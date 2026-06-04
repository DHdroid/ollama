package convert

import (
	"slices"
	"strings"
	"testing"
)

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
	m.ChatTemplate = "{% set test = true %}"

	kv := m.KV(testExaoneTokenizer())
	if got := kv.String("general.architecture"); got != "exaone4" {
		t.Fatalf("architecture = %q, want exaone4", got)
	}
	if _, ok := kv["tokenizer.ggml.scores"]; ok {
		t.Fatal("tokenizer scores should be omitted for EXAONE 4.5")
	}
	if _, ok := kv["exaone4.attention.key_length"]; ok {
		t.Fatal("key_length should be omitted for EXAONE 4.5")
	}
	if got := kv.Uint("block_count"); got != 65 {
		t.Fatalf("block_count = %d, want 65", got)
	}
	if got := kv.Uint("nextn_predict_layers"); got != 1 {
		t.Fatalf("nextn_predict_layers = %d, want 1", got)
	}
	if _, ok := kv["exaone4.vision.block_count"]; ok {
		t.Fatal("text KV should not contain vision metadata")
	}

	tokenizer := testExaoneTokenizer()
	m.adjustTokenizer(tokenizer)
	if got := tokenizer.Pre; got != "exaone-moe" {
		t.Fatalf("tokenizer pre = %q, want exaone-moe", got)
	}
	if got := tokenizer.Template; got != "{% set test = true %}" {
		t.Fatalf("tokenizer template = %q, want chat template", got)
	}

	projectorKV := m.ProjectorKV(testExaoneTokenizer())
	if got := projectorKV.String("general.architecture"); got != "clip" {
		t.Fatalf("projector architecture = %q, want clip", got)
	}
	if got := projectorKV.String("general.type"); got != "mmproj" {
		t.Fatalf("projector type = %q, want mmproj", got)
	}
	if got := projectorKV["clip.projector_type"]; got != "exaone4_5" {
		t.Fatalf("clip.projector_type = %q, want exaone4_5", got)
	}
	if got := projectorKV["general.file_type"]; got != uint32(32) {
		t.Fatalf("projector file type = %d, want 32", got)
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
	if got := projectorKV["clip.vision.image_min_pixels"]; got != uint32(3136) {
		t.Fatalf("clip.vision.image_min_pixels = %d, want 3136", got)
	}
	if got := projectorKV["clip.vision.image_max_pixels"]; got != uint32(3211264) {
		t.Fatalf("clip.vision.image_max_pixels = %d, want 3211264", got)
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
	qkvWeight := &fakeTensor{name: "v.blk.0.attn_qkv.weight", shape: []uint64{3072, 2048}, data: slices.Repeat([]float32{1}, 3072*2048), sourceDType: "BF16", kind: tensorKindFP16}
	qkvBias := &fakeTensor{name: "v.blk.0.attn_qkv.bias", shape: []uint64{3072}, data: slices.Repeat([]float32{1}, 3072)}
	merger := &fakeTensor{name: "v.merger.mlp.2.weight", shape: []uint64{8192, 2048}, data: slices.Repeat([]float32{1}, 8192*2048)}
	postLN := &fakeTensor{name: "v.merger.ln_q.weight", shape: []uint64{2048}, data: slices.Repeat([]float32{1}, 2048)}
	mtp := &fakeTensor{name: "mtp.layers.0.self_attn.q_proj.weight", shape: []uint64{2, 2}, data: slices.Repeat([]float32{1}, 4)}
	mtpFC := &fakeTensor{name: "mtp.fc.weight", shape: []uint64{4, 2}, data: slices.Repeat([]float32{1}, 8)}
	text := m.TextTensors([]Tensor{patch, qkvWeight, mtp}, testExaoneTokenizer())
	textNames := make(map[string][]uint64)
	for _, tt := range text {
		textNames[tt.Name] = tt.Shape
	}
	if _, ok := textNames["v.patch_embd.weight"]; ok {
		t.Fatal("text tensors included vision patch tensor")
	}
	if _, ok := textNames["blk.64.self_attn.q_proj.weight"]; !ok {
		t.Fatal("mtp layer tensor was not emitted as final nextn block")
	}
	text = m.TextTensors([]Tensor{mtpFC}, testExaoneTokenizer())
	if len(text) != 1 || text[0].Name != "blk.64.nextn.eh_proj.weight" {
		t.Fatalf("mtp fc tensor = %#v, want blk.64.nextn.eh_proj.weight", text)
	}

	projector := m.ProjectorTensors([]Tensor{patch, qkvWeight, qkvBias, merger, postLN, mtp})
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
	if got := projectorNames["v.blk.0.attn_qkv.weight"]; !slices.Equal(got, []uint64{3072, 2048}) {
		t.Fatalf("attn_qkv weight shape = %v, want [3072 2048]", got)
	}
	for _, tt := range projector {
		if tt.Name == "v.blk.0.attn_qkv.weight" && tt.Kind != tensorKindBF16 {
			t.Fatalf("attn_qkv weight kind = %v, want bf16", tt.Kind)
		}
	}
	if got := projectorNames["v.blk.0.attn_qkv.bias"]; !slices.Equal(got, []uint64{3072}) {
		t.Fatalf("attn_qkv bias shape = %v, want [3072]", got)
	}
	if got := projectorNames["mm.2.weight"]; !slices.Equal(got, []uint64{8192, 2048}) {
		t.Fatalf("mm.2.weight shape = %v, want [8192 2048]", got)
	}
	if got := projectorNames["v.post_ln.weight"]; !slices.Equal(got, []uint64{2048}) {
		t.Fatalf("v.post_ln.weight shape = %v, want [2048]", got)
	}
	if _, ok := projectorNames["mtp.layers.0.self_attn.q_proj.weight"]; ok {
		t.Fatal("projector tensors included MTP tensor")
	}
}
