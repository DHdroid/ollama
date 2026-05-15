package exaone4

import (
	"testing"
)

func TestParseConfigExaone4TopLevel(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"architectures": ["Exaone4ForCausalLM"],
		"model_type": "exaone4",
		"hidden_size": 2048,
		"intermediate_size": 4096,
		"num_hidden_layers": 30,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"max_position_embeddings": 65536,
		"rms_norm_eps": 1e-5,
		"sliding_window_pattern": "LLLG",
		"rope_theta": 1000000,
		"vocab_size": 102400
	}`))
	if err != nil {
		t.Fatal(err)
	}

	if cfg.ModelType != "exaone4" {
		t.Fatalf("ModelType = %q, want exaone4", cfg.ModelType)
	}
	if cfg.HiddenSize != 2048 || cfg.HeadDim != 64 {
		t.Fatalf("HiddenSize/HeadDim = %d/%d, want 2048/64", cfg.HiddenSize, cfg.HeadDim)
	}
	if cfg.NumHiddenLayers != 30 || cfg.NumAttentionHeads != 32 || cfg.NumKeyValueHeads != 8 {
		t.Fatalf("layer/head config = %d/%d/%d, want 30/32/8", cfg.NumHiddenLayers, cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}
	if cfg.SlidingWindowPattern != "LLLG" {
		t.Fatalf("SlidingWindowPattern = %q, want LLLG", cfg.SlidingWindowPattern)
	}
	if cfg.RopeTheta != 1000000 {
		t.Fatalf("RopeTheta = %v, want 1000000", cfg.RopeTheta)
	}
}

func TestParseConfigExaone4RopeParameters(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
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
