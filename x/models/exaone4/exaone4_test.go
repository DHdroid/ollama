package exaone4

import (
	"testing"
)

func TestParseConfigExaone45TextConfig(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
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
	if cfg.RopeTheta != 1000000 {
		t.Fatalf("RopeTheta = %v, want 1000000", cfg.RopeTheta)
	}
}

func TestParseConfigExaone45RopeParameters(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"text_config": {
			"model_type": "exaone4",
			"hidden_size": 2048,
			"intermediate_size": 4096,
			"num_hidden_layers": 30,
			"num_attention_heads": 32,
			"num_key_value_heads": 8,
			"max_position_embeddings": 128000,
			"rms_norm_eps": 1e-5,
			"rope_parameters": {
				"factor": 16.0,
				"high_freq_factor": 4.0,
				"low_freq_factor": 1.0,
				"original_max_position_embeddings": 8192,
				"rope_type": "llama3",
				"rope_theta": 1000000.0
			}
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.RopeParameters == nil {
		t.Fatal("RopeParameters was not populated")
	}
	if cfg.RopeTheta != 1000000 {
		t.Fatalf("RopeTheta = %v, want 1000000", cfg.RopeTheta)
	}
	if cfg.RopeFreqs == nil {
		t.Fatal("RopeFreqs was not built from rope_parameters")
	}
}

func TestIsLayerSliding(t *testing.T) {
	cfg := &Config{
		NumHiddenLayers:      8,
		SlidingWindowPattern: "LLLG",
	}
	for i, want := range []bool{true, true, true, false, true, true, true, false} {
		if got := isLayerSliding(int32(i), cfg); got != want {
			t.Fatalf("isLayerSliding(%d) = %v, want %v", i, got, want)
		}
	}

	cfg.LayerTypes = []string{
		"sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
		"sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
	}
	for i, want := range []bool{true, true, true, false, true, true, true, false} {
		if got := isLayerSliding(int32(i), cfg); got != want {
			t.Fatalf("isLayerSliding with layer_types (%d) = %v, want %v", i, got, want)
		}
	}
}
