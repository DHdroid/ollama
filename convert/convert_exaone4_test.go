package convert

import (
	"strings"
	"testing"
)

func testExaoneTokenizer() *Tokenizer {
	return &Tokenizer{Vocabulary: &Vocabulary{Model: "exaone", Tokens: []string{"a"}, Scores: []float32{0}, Types: []int32{0}}, Pre: "exaone"}
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

	kv := m.KV(testExaoneTokenizer())
	if got := kv.String("general.architecture"); got != "exaone4" {
		t.Fatalf("architecture = %q, want exaone4", got)
	}
	if _, ok := kv["tokenizer.ggml.scores"]; ok {
		t.Fatal("tokenizer scores should be omitted for EXAONE 4")
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

	tokenizer := testExaoneTokenizer()
	m.ChatTemplate = "{% set exaone4 = true %}"
	m.adjustTokenizer(tokenizer)
	if got := tokenizer.Pre; got != "exaone4" {
		t.Fatalf("tokenizer pre = %q, want exaone4", got)
	}
	if got := tokenizer.Template; got != "{% set exaone4 = true %}" {
		t.Fatalf("tokenizer template = %q, want chat template", got)
	}
}

func TestExaone4SlidingWindowPattern(t *testing.T) {
	slidingWindow := uint32(4096)
	m := &exaone4Model{
		ModelParameters:       ModelParameters{VocabSize: 1000},
		MaxPositionEmbeddings: 32768,
		HiddenSize:            2048,
		HiddenLayers:          8,
		IntermediateSize:      4096,
		NumAttentionHeads:     32,
		NumKeyValueHeads:      8,
		RopeTheta:             1_000_000,
		RMSNormEPS:            1e-5,
		SlidingWindow:         &slidingWindow,
		SlidingWindowPattern:  "LLLG",
	}

	kv := m.KV(testExaoneTokenizer())
	if got := kv["exaone4.attention.sliding_window"]; got != uint32(4096) {
		t.Fatalf("sliding window = %v, want 4096", got)
	}
	want := []bool{true, true, true, false, true, true, true, false}
	got, ok := kv["exaone4.attention.sliding_window_pattern"].([]bool)
	if !ok {
		t.Fatalf("sliding window pattern type = %T, want []bool", kv["exaone4.attention.sliding_window_pattern"])
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("sliding window pattern = %v, want %v", got, want)
		}
	}
}
