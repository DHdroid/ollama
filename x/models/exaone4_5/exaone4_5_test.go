package exaone4_5

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/exaone4"
)

var _ base.Model = (*Model)(nil)
var _ base.MTPEmbeddingModel = (*Model)(nil)
var _ base.DraftModel = (*MTPModel)(nil)
var _ base.MTPDraftModel = (*MTPModel)(nil)
var _ base.CachedMTPDraftModel = (*MTPModel)(nil)

func TestParseTextConfig(t *testing.T) {
	cfg, err := parseTextConfig([]byte(`{
		"architectures": ["Exaone4_5_ForConditionalGeneration"],
		"model_type": "exaone4_5",
		"tie_word_embeddings": false,
		"vocab_size": 153600,
		"text_config": {
			"architectures": ["Exaone4ForCausalLM"],
			"model_type": "exaone4_5_text",
			"hidden_size": 5120,
			"intermediate_size": 27392,
			"num_hidden_layers": 64,
			"num_attention_heads": 40,
			"num_key_value_heads": 8,
			"max_position_embeddings": 262144,
			"rms_norm_eps": 1e-5,
			"sliding_window": 4096,
			"sliding_window_pattern": "LLLG",
			"layer_types": [
				"sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
			],
			"rope_theta": 1000000
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}

	if cfg.ModelType != "exaone4_5_text" {
		t.Fatalf("ModelType = %q, want exaone4_5_text", cfg.ModelType)
	}
	if cfg.HiddenSize != 5120 || cfg.HeadDim != 128 {
		t.Fatalf("HiddenSize/HeadDim = %d/%d, want 5120/128", cfg.HiddenSize, cfg.HeadDim)
	}
	if cfg.NumHiddenLayers != 64 || cfg.NumAttentionHeads != 40 || cfg.NumKeyValueHeads != 8 {
		t.Fatalf("layer/head config = %d/%d/%d, want 64/40/8", cfg.NumHiddenLayers, cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}
	if cfg.SlidingWindow != 4096 || cfg.SlidingWindowPattern != "LLLG" {
		t.Fatalf("sliding config = %d/%q, want 4096/LLLG", cfg.SlidingWindow, cfg.SlidingWindowPattern)
	}
	if cfg.TieWordEmbeddings {
		t.Fatal("TieWordEmbeddings = true, want false from envelope override")
	}
	if cfg.VocabSize != 153600 {
		t.Fatalf("VocabSize = %d, want 153600", cfg.VocabSize)
	}
	if cfg.RopeTheta != 1000000 {
		t.Fatalf("RopeTheta = %v, want 1000000", cfg.RopeTheta)
	}
}

func TestMTPDraftDefaults(t *testing.T) {
	defaults := (&Model{}).MTPDraftDefaults(false)
	if !defaults.Enabled || defaults.InitialDraftTokens != 3 || defaults.MaxDraftTokens != 3 {
		t.Fatalf("MTPDraftDefaults = %+v, want enabled initial=3 max=3", defaults)
	}
}

func TestMTPLayerSliding(t *testing.T) {
	if !isMTPLayerSliding(&exaone4.Config{
		LayerTypes: []string{"sliding_attention", "full_attention"},
	}) {
		t.Fatal("isMTPLayerSliding with sliding layer_types = false, want true")
	}
	if !isMTPLayerSliding(&exaone4.Config{
		SlidingWindowPattern: "LLLG",
	}) {
		t.Fatal("isMTPLayerSliding with LLLG pattern = false, want true")
	}
	if isMTPLayerSliding(&exaone4.Config{
		LayerTypes: []string{"full_attention", "sliding_attention"},
	}) {
		t.Fatal("isMTPLayerSliding with full first layer = true, want false")
	}
}
