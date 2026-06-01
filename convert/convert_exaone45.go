package convert

import (
	"cmp"
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type exaone45Model struct {
	exaone4Model `json:"text_config"`
	VisionModel  struct {
		Depth             uint32  `json:"depth"`
		HiddenSize        uint32  `json:"hidden_size"`
		IntermediateSize  uint32  `json:"intermediate_size"`
		NumHeads          uint32  `json:"num_heads"`
		NumKeyValueHeads  uint32  `json:"num_key_value_heads"`
		InChannels        uint32  `json:"in_channels"`
		PatchSize         uint32  `json:"patch_size"`
		SpatialMergeSize  uint32  `json:"spatial_merge_size"`
		TemporalPatchSize uint32  `json:"temporal_patch_size"`
		WindowSize        uint32  `json:"window_size"`
		RMSNormEPS        float32 `json:"rms_norm_eps"`
		RopeTheta         float32 `json:"rope_theta"`
		ImageSize         uint32  `json:"image_size"`
		FullAttnBlocks    []int32 `json:"fullatt_block_indexes"`
	} `json:"vision_config"`
	Preprocessor struct {
		MinPixels         uint32    `json:"min_pixels"`
		MaxPixels         uint32    `json:"max_pixels"`
		ImageMean         []float32 `json:"image_mean"`
		ImageStd          []float32 `json:"image_std"`
		TemporalPatchSize uint32    `json:"temporal_patch_size"`
		Size              struct {
			ShortestEdge uint32 `json:"shortest_edge"`
			LongestEdge  uint32 `json:"longest_edge"`
		} `json:"size"`
	} `json:"-"`
}

var _ ModelConverter = (*exaone45Model)(nil)
var _ MultimodalConverter = (*exaone45Model)(nil)
var _ moreParser = (*exaone45Model)(nil)

func (m *exaone45Model) architecture() string {
	return "exaone4_5"
}

func (m *exaone45Model) parseMore(fsys fs.FS) error {
	bts, err := fs.ReadFile(fsys, "preprocessor_config.json")
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil
		}
		return err
	}
	return json.Unmarshal(bts, &m.Preprocessor)
}

func (m *exaone45Model) KV(t *Tokenizer) KV {
	return m.TextKV(t)
}

func (m *exaone45Model) TextKV(t *Tokenizer) KV {
	kv := m.exaone4Model.KV(t)
	arch := m.architecture()
	kv["general.architecture"] = arch
	moveArchKV(kv, "exaone4", arch)
	return kv
}

func (m *exaone45Model) ProjectorKV(*Tokenizer) KV {
	vision := m.VisionModel
	kv := KV{
		"general.architecture":                     "clip",
		"general.type":                             "mmproj",
		"general.file_type":                        uint32(1),
		"general.quantization_version":             uint32(2),
		"clip.has_vision_encoder":                  true,
		"clip.projector_type":                      "exaone4_5",
		"clip.use_silu":                            true,
		"clip.vision.projection_dim":               m.HiddenSize,
		"clip.vision.image_size":                   cmp.Or(vision.ImageSize, uint32(560)),
		"clip.vision.patch_size":                   cmp.Or(vision.PatchSize, uint32(14)),
		"clip.vision.embedding_length":             vision.HiddenSize,
		"clip.vision.feed_forward_length":          vision.IntermediateSize,
		"clip.vision.block_count":                  vision.Depth,
		"clip.vision.attention.head_count":         vision.NumHeads,
		"clip.vision.attention.head_count_kv":      vision.NumKeyValueHeads,
		"clip.vision.attention.layer_norm_epsilon": cmp.Or(vision.RMSNormEPS, m.RMSNormEPS, float32(1e-6)),
		"clip.vision.window_size":                  cmp.Or(vision.WindowSize, uint32(112)),
		"clip.vision.n_wa_pattern":                 m.visionWindowAttentionPattern(),
	}
	if len(m.Preprocessor.ImageMean) == 3 {
		kv["clip.vision.image_mean"] = m.Preprocessor.ImageMean
	}
	if len(m.Preprocessor.ImageStd) == 3 {
		kv["clip.vision.image_std"] = m.Preprocessor.ImageStd
	}
	return kv
}

func (m *exaone45Model) visionWindowAttentionPattern() uint32 {
	if len(m.VisionModel.FullAttnBlocks) > 0 && m.VisionModel.FullAttnBlocks[0] >= 0 {
		return uint32(m.VisionModel.FullAttnBlocks[0] + 1)
	}
	return 7
}

func moveArchKV(kv KV, oldArch, newArch string) {
	oldPrefix := oldArch + "."
	for _, key := range slices.Sorted(kv.Keys()) {
		if !strings.HasPrefix(key, oldPrefix) {
			continue
		}
		value := kv[key]
		delete(kv, key)
		kv[newArch+"."+strings.TrimPrefix(key, oldPrefix)] = value
	}
}

func (m *exaone45Model) Tensors(ts []Tensor) []*ggml.Tensor {
	out := m.TextTensors(ts, nil)
	return append(out, m.ProjectorTensors(ts)...)
}

func (m *exaone45Model) TextTensors(ts []Tensor, _ *Tokenizer) []*ggml.Tensor {
	var out []*ggml.Tensor
	var rest []Tensor
	for _, t := range ts {
		switch {
		case strings.HasPrefix(t.Name(), "mtp."):
			out = append(out, &ggml.Tensor{Name: restoreExaoneMTPName(t.Name()), Kind: t.Kind(), Shape: t.Shape(), WriterTo: t})
		case exaone45VisionTensor(t.Name()):
			continue
		default:
			rest = append(rest, t)
		}
	}
	return append(m.exaone4Model.Tensors(rest), out...)
}

func exaone45VisionTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.")
}

func (m *exaone45Model) ProjectorTensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor
	for _, t := range ts {
		name := m.projectorTensorName(t.Name())
		if !exaone45VisionTensor(name) {
			continue
		}

		switch {
		case name == "v.patch_embd.weight" && len(t.Shape()) == 5:
			out = append(out, exaone45PatchEmbedTensors(t)...)
		case strings.Contains(name, "attn_qkv"):
			out = append(out, m.splitVisionQKV(t, name)...)
		default:
			kind := t.Kind()
			var writer io.WriterTo = t
			if sourceDType(t) == "BF16" && kind == tensorKindFP16 {
				kind = tensorKindBF16
				writer = tensorBF16Writer{tensor: t}
			}
			out = append(out, &ggml.Tensor{Name: name, Kind: kind, Shape: slices.Clone(t.Shape()), WriterTo: writer})
		}
	}
	return out
}

func (m *exaone45Model) projectorTensorName(name string) string {
	if strings.HasPrefix(name, "v.merger.") {
		name = strings.Replace(name, "v.merger.ln_q", "mm.input_norm", 1)
		name = strings.Replace(name, "v.merger.mlp.0", "mm.0", 1)
		name = strings.Replace(name, "v.merger.mlp.2", "mm.2", 1)
	}
	return name
}

func exaone45PatchEmbedTensors(t Tensor) []*ggml.Tensor {
	shape := t.Shape()
	if len(shape) != 5 || shape[2] != 2 {
		return nil
	}
	outShape := []uint64{shape[0], shape[1], shape[3], shape[4]}
	return []*ggml.Tensor{
		{
			Name:     "v.patch_embd.weight",
			Kind:     tensorKindFP32,
			Shape:    slices.Clone(outShape),
			WriterTo: tensorFloat32Writer{tensor: t, repacker: qwenTemporalPatchEmbedSlice(0)},
		},
		{
			Name:     "v.patch_embd.weight.1",
			Kind:     tensorKindFP32,
			Shape:    slices.Clone(outShape),
			WriterTo: tensorFloat32Writer{tensor: t, repacker: qwenTemporalPatchEmbedSlice(1)},
		},
	}
}

func (m *exaone45Model) splitVisionQKV(t Tensor, name string) []*ggml.Tensor {
	hiddenSize := cmp.Or(m.VisionModel.HiddenSize, uint32(0))
	numHeads := cmp.Or(m.VisionModel.NumHeads, uint32(1))
	numKVHeads := cmp.Or(m.VisionModel.NumKeyValueHeads, numHeads)
	kvSize := hiddenSize * numKVHeads / numHeads
	if hiddenSize == 0 || kvSize == 0 {
		return slices.Collect(splitDim(t, 0,
			split{Replacer: strings.NewReplacer("attn_qkv", "attn_q")},
			split{Replacer: strings.NewReplacer("attn_qkv", "attn_k")},
			split{Replacer: strings.NewReplacer("attn_qkv", "attn_v")},
		))
	}
	return slices.Collect(splitDim(t, 0,
		split{Replacer: strings.NewReplacer("attn_qkv", "attn_q"), dim: int(hiddenSize)},
		split{Replacer: strings.NewReplacer("attn_qkv", "attn_k"), dim: int(kvSize)},
		split{Replacer: strings.NewReplacer("attn_qkv", "attn_v"), dim: int(kvSize)},
	))
}

func restoreExaoneMTPName(name string) string {
	replacer := strings.NewReplacer(
		"attn_k_norm", "self_attn.k_norm",
		"attn_q_norm", "self_attn.q_norm",
		"attn_output", "self_attn.o_proj",
		"attn_k", "self_attn.k_proj",
		"attn_q", "self_attn.q_proj",
		"attn_v", "self_attn.v_proj",
		"ffn_down", "mlp.down_proj",
		"ffn_gate", "mlp.gate_proj",
		"ffn_up", "mlp.up_proj",
		"post_attention_norm", "post_attention_layernorm",
		"post_ffw_norm", "post_feedforward_layernorm",
	)
	return replacer.Replace(name)
}

func (m *exaone45Model) Replacements() []string {
	replacements := []string{
		"model.language_", "",
		"model.visual", "v",
		"visual", "v",
		"patch_embed.proj", "patch_embd",
		"blocks", "blk",
		"attn.qkv", "attn_qkv",
		"attn.proj", "attn_out",
		"norm1", "ln1",
		"norm2", "ln2",
		"merger.ln_q", "merger.ln_q",
		"merger.mlp.0", "merger.mlp.0",
		"merger.mlp.2", "merger.mlp.2",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"mlp.down_proj", "ffn_down",
	}
	return append(m.exaone4Model.Replacements(), replacements...)
}
