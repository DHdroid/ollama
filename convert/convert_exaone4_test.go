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

func TestExaone45PreservesMTPAndMapsVision(t *testing.T) {
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
	if _, ok := kv["exaone4_5.nextn_predict_layers"]; ok {
		t.Fatal("nextn_predict_layers should not be emitted when mtp.* tensors are preserved")
	}
	if got := kv.Uint("vision.block_count"); got != 28 {
		t.Fatalf("vision.block_count = %d, want 28", got)
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
	mtp := &fakeTensor{name: "mtp.layers.0.attn_q.weight", shape: []uint64{2, 2}, data: slices.Repeat([]float32{1}, 4)}
	out := m.Tensors([]Tensor{patch, mtp})
	names := make(map[string][]uint64)
	for _, tt := range out {
		names[tt.Name] = tt.Shape
	}
	if got := names["v.patch_embd_0.weight"]; !slices.Equal(got, []uint64{2, 3, 2, 2}) {
		t.Fatalf("patch_embd_0 shape = %v, want [2 3 2 2]", got)
	}
	if got := names["v.patch_embd_1.weight"]; !slices.Equal(got, []uint64{2, 3, 2, 2}) {
		t.Fatalf("patch_embd_1 shape = %v, want [2 3 2 2]", got)
	}
	if _, ok := names["mtp.layers.0.self_attn.q_proj.weight"]; !ok {
		t.Fatal("mtp layer tensor was not restored to HF name")
	}
}
