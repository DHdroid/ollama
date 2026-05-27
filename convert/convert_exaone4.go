package convert

import (
	"cmp"
	"math"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type exaone4Model struct {
	ModelParameters
	MaxPositionEmbeddings uint32     `json:"max_position_embeddings"`
	HiddenSize            uint32     `json:"hidden_size"`
	HiddenLayers          uint32     `json:"num_hidden_layers"`
	IntermediateSize      uint32     `json:"intermediate_size"`
	NumAttentionHeads     uint32     `json:"num_attention_heads"`
	NumKeyValueHeads      uint32     `json:"num_key_value_heads"`
	HeadDim               uint32     `json:"head_dim"`
	RopeTheta             float32    `json:"rope_theta"`
	RMSNormEPS            float32    `json:"rms_norm_eps"`
	SlidingWindow         *uint32    `json:"sliding_window"`
	SlidingWindowPattern  any        `json:"sliding_window_pattern"`
	LayerTypes            []string   `json:"layer_types"`
	RopeParameters        ropeParams `json:"rope_parameters"`
	RopeScaling           ropeParams `json:"rope_scaling"`
}

type ropeParams struct {
	Type                          string  `json:"rope_type"`
	Theta                         float32 `json:"rope_theta"`
	Factor                        float32 `json:"factor"`
	LowFreqFactor                 float32 `json:"low_freq_factor"`
	HighFreqFactor                float32 `json:"high_freq_factor"`
	OriginalMaxPositionEmbeddings uint32  `json:"original_max_position_embeddings"`
}

var _ ModelConverter = (*exaone4Model)(nil)

func (m *exaone4Model) architecture() string {
	return "exaone4"
}

func (m *exaone4Model) KV(t *Tokenizer) KV {
	kv := m.ModelParameters.KV(t)
	arch := m.architecture()
	kv["general.architecture"] = arch
	kv[arch+".vocab_size"] = cmp.Or(m.VocabSize, m.TextModel.VocabSize)
	kv[arch+".block_count"] = m.HiddenLayers
	kv[arch+".context_length"] = m.MaxPositionEmbeddings
	kv[arch+".embedding_length"] = m.HiddenSize
	kv[arch+".feed_forward_length"] = m.IntermediateSize
	kv[arch+".attention.head_count"] = m.NumAttentionHeads
	kv[arch+".attention.head_count_kv"] = m.NumKeyValueHeads
	if headDim := m.headDim(); headDim > 0 {
		kv[arch+".attention.key_length"] = headDim
		kv[arch+".attention.value_length"] = headDim
	}
	kv[arch+".rope.freq_base"] = cmp.Or(m.ropeParams().Theta, m.RopeTheta, float32(10000))
	kv[arch+".attention.layer_norm_rms_epsilon"] = m.RMSNormEPS
	if m.SlidingWindow != nil {
		kv[arch+".attention.sliding_window"] = *m.SlidingWindow
	}
	if pattern := m.slidingWindowPattern(); len(pattern) > 0 {
		kv[arch+".attention.sliding_window_pattern"] = pattern
	}
	return kv
}

func (m *exaone4Model) ropeParams() ropeParams {
	if m.RopeParameters.Type != "" || m.RopeParameters.Theta != 0 {
		return m.RopeParameters
	}
	return m.RopeScaling
}

func (m *exaone4Model) headDim() uint32 {
	if m.HeadDim > 0 {
		return m.HeadDim
	}
	if m.HiddenSize > 0 && m.NumAttentionHeads > 0 {
		return m.HiddenSize / m.NumAttentionHeads
	}
	return 0
}

func (m *exaone4Model) slidingWindowPattern() []bool {
	switch pattern := m.SlidingWindowPattern.(type) {
	case []bool:
		if len(pattern) > 0 {
			return pattern
		}
	case string:
		if pattern != "" && m.HiddenLayers > 0 {
			out := make([]bool, m.HiddenLayers)
			for i := range out {
				out[i] = pattern[i%len(pattern)] == 'L'
			}
			return out
		}
	case float64:
		if pattern > 0 && m.HiddenLayers > 0 {
			interval := int(pattern)
			out := make([]bool, m.HiddenLayers)
			for i := range out {
				out[i] = (i+1)%interval != 0
			}
			return out
		}
	}
	if len(m.LayerTypes) == 0 {
		return nil
	}
	out := make([]bool, len(m.LayerTypes))
	for i, layerType := range m.LayerTypes {
		out[i] = layerType == "sliding_attention"
	}
	return out
}

func (m *exaone4Model) ropeFactors() ropeFactor {
	if strings.ToLower(m.ropeParams().Type) != "llama3" {
		return nil
	}

	headDim := int(m.headDim())
	if headDim <= 0 {
		return nil
	}
	base := float64(cmp.Or(m.ropeParams().Theta, m.RopeTheta, float32(10000)))
	factor := float64(cmp.Or(m.ropeParams().Factor, float32(16)))
	lowFreqFactor := float64(cmp.Or(m.ropeParams().LowFreqFactor, float32(1)))
	highFreqFactor := float64(cmp.Or(m.ropeParams().HighFreqFactor, float32(4)))
	oldContextLen := float64(cmp.Or(m.ropeParams().OriginalMaxPositionEmbeddings, uint32(8192)))

	lowFreqWavelen := oldContextLen / lowFreqFactor
	highFreqWavelen := oldContextLen / highFreqFactor
	factors := make(ropeFactor, 0, headDim/2)
	for i := 0; i < headDim; i += 2 {
		freq := 1 / math.Pow(base, float64(i)/float64(headDim))
		wavelen := 2 * math.Pi / freq
		scale := float64(1)
		switch {
		case wavelen < highFreqWavelen:
			scale = 1
		case wavelen > lowFreqWavelen:
			scale = factor
		default:
			smooth := (oldContextLen/wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
			scale = 1 / ((1-smooth)/factor + smooth)
		}
		factors = append(factors, float32(scale))
	}
	return factors
}

func (m *exaone4Model) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor
	if factors := m.ropeFactors(); len(factors) > 0 {
		out = append(out, &ggml.Tensor{
			Name:     "rope_freqs.weight",
			Kind:     tensorKindFP32,
			Shape:    []uint64{uint64(len(factors))},
			WriterTo: factors,
		})
	}
	for _, t := range ts {
		out = append(out, &ggml.Tensor{Name: t.Name(), Kind: t.Kind(), Shape: t.Shape(), WriterTo: t})
	}
	return out
}

func (m *exaone4Model) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.layers", "blk",
		"model.norm", "output_norm",
		"self_attn.k_proj", "attn_k",
		"self_attn.k_norm", "attn_k_norm",
		"self_attn.v_proj", "attn_v",
		"self_attn.q_proj", "attn_q",
		"self_attn.q_norm", "attn_q_norm",
		"self_attn.o_proj", "attn_output",
		"mlp.down_proj", "ffn_down",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"post_attention_layernorm", "post_attention_norm",
		"post_feedforward_layernorm", "post_ffw_norm",
	}
}
