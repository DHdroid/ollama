package convert

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type exaone45Model struct {
	exaone4Model          `json:"text_config"`
	NumNextNPredictLayers uint32 `json:"num_nextn_predict_layers"`
	VisionModel           struct {
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
	ChatTemplate string `json:"-"`
}

var (
	_ MultimodalConverter = (*exaone45Model)(nil)
	_ moreParser          = (*exaone45Model)(nil)
	_ tokenizerAdjuster   = (*exaone45Model)(nil)
)

func (m *exaone45Model) parseMore(fsys fs.FS) error {
	bts, err := fs.ReadFile(fsys, "preprocessor_config.json")
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return m.parseChatTemplate(fsys)
		}
		return err
	}
	if err := json.Unmarshal(bts, &m.Preprocessor); err != nil {
		return err
	}
	return m.parseChatTemplate(fsys)
}

func (m *exaone45Model) parseChatTemplate(fsys fs.FS) error {
	bts, err := fs.ReadFile(fsys, "chat_template.jinja")
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil
		}
		return err
	}
	m.ChatTemplate = string(bts)
	return nil
}

func (m *exaone45Model) TextKV(t *Tokenizer) KV {
	kv := m.exaone4Model.KV(t)
	arch := m.architecture()

	delete(kv, "tokenizer.ggml.scores")
	delete(kv, arch+".attention.key_length")
	delete(kv, arch+".attention.value_length")

	nextn := m.nextnPredictLayers()
	if nextn == 0 {
		return kv
	}

	kv[arch+".block_count"] = m.HiddenLayers + nextn
	kv[arch+".nextn_predict_layers"] = nextn
	if pattern, ok := kv[arch+".attention.sliding_window_pattern"].([]bool); ok {
		for range nextn {
			pattern = append(pattern, true)
		}
		kv[arch+".attention.sliding_window_pattern"] = pattern
	}
	return kv
}

func (m *exaone45Model) KV(t *Tokenizer) KV {
	return m.TextKV(t)
}

func (m *exaone45Model) nextnPredictLayers() uint32 {
	return cmp.Or(m.NumNextNPredictLayers, uint32(1))
}

func (m *exaone45Model) adjustTokenizer(t *Tokenizer) {
	t.Pre = "exaone-moe"
	if m.ChatTemplate != "" {
		t.Template = m.ChatTemplate
	}
}

func (m *exaone45Model) ProjectorKV(*Tokenizer) KV {
	vision := m.VisionModel
	kv := KV{
		"general.architecture":                     "clip",
		"general.type":                             "mmproj",
		"general.file_type":                        uint32(32),
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
		"clip.vision.attention.layer_norm_epsilon": cmp.Or(vision.RMSNormEPS, float32(1e-6)),
		"clip.vision.window_size":                  cmp.Or(vision.WindowSize, uint32(112)),
		"clip.vision.n_wa_pattern":                 m.visionWindowAttentionPattern(),
	}
	if len(m.Preprocessor.ImageMean) == 3 {
		kv["clip.vision.image_mean"] = m.Preprocessor.ImageMean
	}
	if len(m.Preprocessor.ImageStd) == 3 {
		kv["clip.vision.image_std"] = m.Preprocessor.ImageStd
	}
	if m.Preprocessor.MinPixels > 0 {
		kv["clip.vision.image_min_pixels"] = m.Preprocessor.MinPixels
	}
	if m.Preprocessor.MaxPixels > 0 {
		kv["clip.vision.image_max_pixels"] = m.Preprocessor.MaxPixels
	}
	return kv
}

func (m *exaone45Model) visionWindowAttentionPattern() uint32 {
	if len(m.VisionModel.FullAttnBlocks) > 0 && m.VisionModel.FullAttnBlocks[0] >= 0 {
		return uint32(m.VisionModel.FullAttnBlocks[0] + 1)
	}
	return 7
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
			for _, name := range m.mtpTensorNames(t.Name()) {
				out = append(out, &ggml.Tensor{Name: name, Kind: t.Kind(), Shape: t.Shape(), WriterTo: t})
			}
		case exaone45VisionTensor(t.Name()):
			continue
		default:
			rest = append(rest, t)
		}
	}
	return append(m.exaone4Model.Tensors(rest), out...)
}

func (m *exaone45Model) mtpTensorNames(name string) []string {
	base := m.HiddenLayers
	nextn := m.nextnPredictLayers()

	if rest := strings.TrimPrefix(name, "mtp.layers."); rest != name {
		layer, suffix, ok := strings.Cut(rest, ".")
		if !ok {
			return nil
		}
		idx, err := strconv.ParseUint(layer, 10, 32)
		if err != nil || uint32(idx) >= nextn {
			return nil
		}
		return []string{fmt.Sprintf("blk.%d.%s", base+uint32(idx), suffix)}
	}

	var suffix string
	switch name {
	case "mtp.fc.weight":
		suffix = "nextn.eh_proj.weight"
	case "mtp.pre_fc_norm_embedding.weight":
		suffix = "nextn.enorm.weight"
	case "mtp.pre_fc_norm_hidden.weight":
		suffix = "nextn.hnorm.weight"
	case "mtp.norm.weight":
		suffix = "nextn.shared_head_norm.weight"
	default:
		return nil
	}

	names := make([]string, 0, nextn)
	for i := range nextn {
		names = append(names, fmt.Sprintf("blk.%d.%s", base+i, suffix))
	}
	return names
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
		name = strings.Replace(name, "v.merger.ln_q", "v.post_ln", 1)
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
