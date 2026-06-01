package convert

import (
	"slices"
	"strings"
	"testing"
)

func testTokenizer() *Tokenizer {
	return &Tokenizer{Vocabulary: &Vocabulary{Model: "gpt2", Tokens: []string{"a"}, Scores: []float32{0}, Types: []int32{0}}, Pre: "exaone"}
}

func TestExaone4KVAndReplacements(t *testing.T) {
	m := &exaone4Model{
		ModelParameters:       ModelParameters{VocabSize: 1000},
		MaxPositionEmbeddings: 32768,
		HiddenSize:            2048,
		HiddenLayers:          30,
		IntermediateSize:      4096,
		NumAttentionHeads:     32,
		NumKeyValueHeads:      8,
		RopeTheta:             1_000_000,
		RMSNormEPS:            1e-5,
	}
	m.RopeParameters.Type = "llama3"
	m.RopeParameters.Theta = 1_000_000
	m.RopeParameters.Factor = 16
	m.RopeParameters.LowFreqFactor = 1
	m.RopeParameters.HighFreqFactor = 4
	m.RopeParameters.OriginalMaxPositionEmbeddings = 8192

	kv := m.KV(testTokenizer())
	if got := kv.String("general.architecture"); got != "exaone4" {
		t.Fatalf("architecture = %q, want exaone4", got)
	}
	if got := kv.Uint("block_count"); got != 30 {
		t.Fatalf("block_count = %d, want 30", got)
	}
	if got := kv.Uint("attention.key_length"); got != 64 {
		t.Fatalf("key_length = %d, want 64", got)
	}

	replacer := strings.NewReplacer(m.Replacements()...)
	cases := map[string]string{
		"model.embed_tokens.weight":                        "token_embd.weight",
		"model.layers.0.self_attn.q_proj.weight":           "blk.0.attn_q.weight",
		"model.layers.0.self_attn.q_norm.weight":           "blk.0.attn_q_norm.weight",
		"model.layers.0.post_attention_layernorm.weight":   "blk.0.post_attention_norm.weight",
		"model.layers.0.post_feedforward_layernorm.weight": "blk.0.post_ffw_norm.weight",
		"model.layers.0.mlp.down_proj.weight":              "blk.0.ffn_down.weight",
		"model.norm.weight":                                "output_norm.weight",
		"lm_head.weight":                                   "output.weight",
	}
	for in, want := range cases {
		if got := replacer.Replace(in); got != want {
			t.Fatalf("Replace(%q) = %q, want %q", in, got, want)
		}
	}

	out := m.Tensors(nil)
	if len(out) != 1 || out[0].Name != "rope_freqs.weight" || out[0].Shape[0] != 32 {
		t.Fatalf("rope tensor = %#v, want one rope_freqs.weight[32]", out)
	}
}

func TestExaone45SplitsTextAndProjector(t *testing.T) {
	m := &exaone45Model{}
	m.ModelParameters = ModelParameters{VocabSize: 153600}
	m.HiddenSize = 2048
	m.HiddenLayers = 64
	m.IntermediateSize = 8192
	m.NumAttentionHeads = 32
	m.NumKeyValueHeads = 8
	m.MaxPositionEmbeddings = 32768
	m.RopeTheta = 1_000_000
	m.RMSNormEPS = 1e-5
	m.VisionModel.Depth = 28
	m.VisionModel.HiddenSize = 2048
	m.VisionModel.IntermediateSize = 5120
	m.VisionModel.NumHeads = 32
	m.VisionModel.NumKeyValueHeads = 8
	m.VisionModel.InChannels = 3
	m.VisionModel.PatchSize = 14
	m.VisionModel.SpatialMergeSize = 2
	m.VisionModel.TemporalPatchSize = 2
	m.VisionModel.WindowSize = 112
	m.VisionModel.FullAttnBlocks = []int32{6, 13, 20, 27}
	m.Preprocessor.MinPixels = 3136
	m.Preprocessor.MaxPixels = 3211264
	m.Preprocessor.ImageMean = []float32{0.1, 0.2, 0.3}
	m.Preprocessor.ImageStd = []float32{0.4, 0.5, 0.6}

	kv := m.KV(testTokenizer())
	if got := kv.String("general.architecture"); got != "exaone4_5" {
		t.Fatalf("architecture = %q, want exaone4_5", got)
	}
	if got := kv.Uint("block_count"); got != 64 {
		t.Fatalf("block_count = %d, want 64", got)
	}
	if _, ok := kv["exaone4_5.vision.block_count"]; ok {
		t.Fatal("text KV should not contain vision metadata")
	}

	projectorKV := m.ProjectorKV(testTokenizer())
	if got := projectorKV.String("general.architecture"); got != "clip" {
		t.Fatalf("projector architecture = %q, want clip", got)
	}
	if got := projectorKV.String("general.type"); got != "mmproj" {
		t.Fatalf("projector type = %q, want mmproj", got)
	}
	if got := projectorKV["clip.projector_type"]; got != "exaone4_5" {
		t.Fatalf("clip.projector_type = %q, want exaone4_5", got)
	}
	if got := projectorKV["clip.vision.block_count"]; got != uint32(28) {
		t.Fatalf("clip.vision.block_count = %d, want 28", got)
	}
	if got := projectorKV["clip.vision.attention.head_count_kv"]; got != uint32(8) {
		t.Fatalf("clip.vision.attention.head_count_kv = %d, want 8", got)
	}
	if got := projectorKV["clip.vision.n_wa_pattern"]; got != uint32(7) {
		t.Fatalf("clip.vision.n_wa_pattern = %d, want 7", got)
	}

	replacer := strings.NewReplacer(m.Replacements()...)
	cases := map[string]string{
		"model.language_model.layers.0.self_attn.k_proj.weight": "blk.0.attn_k.weight",
		"model.visual.blocks.0.attn.qkv.weight":                 "v.blk.0.attn_qkv.weight",
		"model.visual.blocks.0.norm1.weight":                    "v.blk.0.ln1.weight",
		"model.visual.patch_embed.proj.weight":                  "v.patch_embd.weight",
		"model.visual.merger.mlp.2.weight":                      "v.merger.mlp.2.weight",
	}
	for in, want := range cases {
		if got := replacer.Replace(in); got != want {
			t.Fatalf("Replace(%q) = %q, want %q", in, got, want)
		}
	}

	patch := &fakeTensor{name: "v.patch_embd.weight", shape: []uint64{2, 3, 2, 2, 2}, data: slices.Repeat([]float32{1}, 48)}
	qkvWeight := &fakeTensor{name: "v.blk.0.attn_qkv.weight", shape: []uint64{3072, 2048}, data: slices.Repeat([]float32{1}, 3072*2048)}
	qkvBias := &fakeTensor{name: "v.blk.0.attn_qkv.bias", shape: []uint64{3072}, data: slices.Repeat([]float32{1}, 3072)}
	merger := &fakeTensor{name: "v.merger.mlp.2.weight", shape: []uint64{8192, 2048}, data: slices.Repeat([]float32{1}, 8192*2048)}
	mtp := &fakeTensor{name: "mtp.layers.0.attn_q.weight", shape: []uint64{2, 2}, data: slices.Repeat([]float32{1}, 4)}
	text := m.TextTensors([]Tensor{patch, qkvWeight, mtp}, testTokenizer())
	textNames := make(map[string][]uint64)
	for _, tt := range text {
		textNames[tt.Name] = tt.Shape
	}
	if _, ok := textNames["v.patch_embd.weight"]; ok {
		t.Fatal("text tensors included vision patch tensor")
	}
	if _, ok := textNames["mtp.layers.0.self_attn.q_proj.weight"]; !ok {
		t.Fatal("mtp layer tensor was not restored to HF name")
	}

	projector := m.ProjectorTensors([]Tensor{patch, qkvWeight, qkvBias, merger, mtp})
	projectorNames := make(map[string][]uint64)
	for _, tt := range projector {
		projectorNames[tt.Name] = tt.Shape
	}
	if got := projectorNames["v.patch_embd.weight"]; !slices.Equal(got, []uint64{2, 3, 2, 2}) {
		t.Fatalf("patch_embd.weight shape = %v, want [2 3 2 2]", got)
	}
	if got := projectorNames["v.patch_embd.weight.1"]; !slices.Equal(got, []uint64{2, 3, 2, 2}) {
		t.Fatalf("patch_embd.weight.1 shape = %v, want [2 3 2 2]", got)
	}
	if got := projectorNames["v.blk.0.attn_q.weight"]; !slices.Equal(got, []uint64{2048, 2048}) {
		t.Fatalf("attn_q weight shape = %v, want [2048 2048]", got)
	}
	if got := projectorNames["v.blk.0.attn_k.weight"]; !slices.Equal(got, []uint64{512, 2048}) {
		t.Fatalf("attn_k weight shape = %v, want [512 2048]", got)
	}
	if got := projectorNames["v.blk.0.attn_v.weight"]; !slices.Equal(got, []uint64{512, 2048}) {
		t.Fatalf("attn_v weight shape = %v, want [512 2048]", got)
	}
	if got := projectorNames["v.blk.0.attn_q.bias"]; !slices.Equal(got, []uint64{2048}) {
		t.Fatalf("attn_q bias shape = %v, want [2048]", got)
	}
	if got := projectorNames["mm.2.weight"]; !slices.Equal(got, []uint64{8192, 2048}) {
		t.Fatalf("mm.2.weight shape = %v, want [8192 2048]", got)
	}
	if _, ok := projectorNames["mtp.layers.0.attn_q.weight"]; ok {
		t.Fatal("projector tensors included MTP tensor")
	}
}
