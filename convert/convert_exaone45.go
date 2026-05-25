package convert

import (
	"cmp"
	"encoding/json"
	"errors"
	"io/fs"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/pdevine/tensor"
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
	kv := m.exaone4Model.KV(t)
	arch := m.architecture()
	kv["general.architecture"] = arch
	moveArchKV(kv, "exaone4", arch)

	vision := m.VisionModel
	kv[arch+".vision.projection_dim"] = m.HiddenSize
	kv[arch+".vision.block_count"] = cmp.Or(vision.Depth, uint32(0))
	kv[arch+".vision.embedding_length"] = vision.HiddenSize
	kv[arch+".vision.feed_forward_length"] = vision.IntermediateSize
	kv[arch+".vision.attention.head_count"] = vision.NumHeads
	kv[arch+".vision.attention.head_count_kv"] = vision.NumKeyValueHeads
	kv[arch+".vision.num_channels"] = cmp.Or(vision.InChannels, uint32(3))
	kv[arch+".vision.patch_size"] = vision.PatchSize
	kv[arch+".vision.spatial_merge_size"] = vision.SpatialMergeSize
	kv[arch+".vision.temporal_patch_size"] = cmp.Or(vision.TemporalPatchSize, m.Preprocessor.TemporalPatchSize, uint32(2))
	kv[arch+".vision.window_size"] = vision.WindowSize
	kv[arch+".vision.attention.layer_norm_epsilon"] = cmp.Or(vision.RMSNormEPS, m.RMSNormEPS, float32(1e-6))
	kv[arch+".vision.rope.freq_base"] = cmp.Or(vision.RopeTheta, float32(10000))
	kv[arch+".vision.fullatt_block_indexes"] = vision.FullAttnBlocks
	kv[arch+".vision.min_pixels"] = m.Preprocessor.MinPixels
	kv[arch+".vision.max_pixels"] = m.Preprocessor.MaxPixels
	kv[arch+".vision.shortest_edge"] = m.Preprocessor.Size.ShortestEdge
	kv[arch+".vision.longest_edge"] = m.Preprocessor.Size.LongestEdge
	kv[arch+".vision.image_mean"] = m.Preprocessor.ImageMean
	kv[arch+".vision.image_std"] = m.Preprocessor.ImageStd
	return kv
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
	var out []*ggml.Tensor
	var rest []Tensor
	for _, t := range ts {
		switch {
		case strings.HasPrefix(t.Name(), "mtp."):
			out = append(out, &ggml.Tensor{Name: restoreExaoneMTPName(t.Name()), Kind: t.Kind(), Shape: t.Shape(), WriterTo: t})
		case strings.HasPrefix(t.Name(), "v.patch_embd.weight") && len(t.Shape()) == 5:
			out = append(out, slices.Collect(splitDim(t, 2,
				split{Replacer: strings.NewReplacer("v.patch_embd.weight", "v.patch_embd_0.weight"), afterFunc: squeezeTemporalPatch},
				split{Replacer: strings.NewReplacer("v.patch_embd.weight", "v.patch_embd_1.weight"), afterFunc: squeezeTemporalPatch},
			))...)
			for _, tt := range out[len(out)-2:] {
				shape := t.Shape()
				tt.Shape = []uint64{shape[0], shape[1], shape[3], shape[4]}
			}
		default:
			rest = append(rest, t)
		}
	}
	return append(m.exaone4Model.Tensors(rest), out...)
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

func squeezeTemporalPatch(t tensor.Tensor) (tensor.Tensor, error) {
	shape := t.Shape()
	if len(shape) != 5 || shape[2] != 1 {
		return t, nil
	}
	return t, t.Reshape(shape[0], shape[1], shape[3], shape[4])
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
	}
	return append(m.exaone4Model.Replacements(), replacements...)
}
