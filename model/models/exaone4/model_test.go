package exaone4

import (
	"iter"
	"testing"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/model"
)

type testConfig map[string]any

func (c testConfig) Architecture() string { return "exaone4" }

func (c testConfig) key(key string) string {
	switch {
	case len(key) >= len("tokenizer.") && key[:len("tokenizer.")] == "tokenizer.":
		return key
	case len(key) >= len("general.") && key[:len("general.")] == "general.":
		return key
	default:
		return "exaone4." + key
	}
}

func (c testConfig) String(key string, defaultValue ...string) string {
	if v, ok := c[c.key(key)].(string); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return ""
}

func (c testConfig) Uint(key string, defaultValue ...uint32) uint32 {
	switch v := c[c.key(key)].(type) {
	case uint32:
		return v
	case int:
		return uint32(v)
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return 0
}

func (c testConfig) Float(key string, defaultValue ...float32) float32 {
	if v, ok := c[c.key(key)].(float32); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return 0
}

func (c testConfig) Bool(key string, defaultValue ...bool) bool {
	if v, ok := c[c.key(key)].(bool); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return false
}

func (c testConfig) Strings(key string, defaultValue ...[]string) []string {
	if v, ok := c[c.key(key)].([]string); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return nil
}

func (c testConfig) Ints(key string, defaultValue ...[]int32) []int32 {
	if v, ok := c[c.key(key)].([]int32); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return nil
}

func (c testConfig) Uints(key string, defaultValue ...[]uint32) []uint32 {
	if v, ok := c[c.key(key)].([]uint32); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return nil
}

func (c testConfig) Floats(key string, defaultValue ...[]float32) []float32 {
	if v, ok := c[c.key(key)].([]float32); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return nil
}

func (c testConfig) Bools(key string, defaultValue ...[]bool) []bool {
	if v, ok := c[c.key(key)].([]bool); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return nil
}

func (c testConfig) Len() int { return len(c) }

func (c testConfig) Keys() iter.Seq[string] {
	return func(yield func(string) bool) {
		for key := range c {
			if !yield(key) {
				return
			}
		}
	}
}

func (c testConfig) Value(key string) any { return c[key] }

func baseConfig() testConfig {
	return testConfig{
		"general.architecture":                     "exaone4",
		"tokenizer.ggml.model":                     "gpt2",
		"tokenizer.ggml.pre":                       "exaone4",
		"tokenizer.ggml.tokens":                    []string{"[PAD]", "[BOS]", "[EOS]", "a"},
		"tokenizer.ggml.token_type":                []int32{3, 3, 3, 1},
		"tokenizer.ggml.merges":                    []string{},
		"tokenizer.ggml.bos_token_id":              uint32(1),
		"tokenizer.ggml.eos_token_id":              uint32(2),
		"exaone4.block_count":                      uint32(4),
		"exaone4.embedding_length":                 uint32(128),
		"exaone4.attention.head_count":             uint32(8),
		"exaone4.attention.head_count_kv":          uint32(2),
		"exaone4.attention.key_length":             uint32(16),
		"exaone4.attention.value_length":           uint32(16),
		"exaone4.attention.layer_norm_rms_epsilon": float32(1e-5),
		"exaone4.rope.freq_base":                   float32(1000000),
	}
}

func TestNewCausalConfig(t *testing.T) {
	mm, err := New(baseConfig())
	if err != nil {
		t.Fatal(err)
	}

	m := mm.(*Model)
	if _, ok := m.Cache.(*kvcache.Causal); !ok {
		t.Fatalf("cache = %T, want *kvcache.Causal", m.Cache)
	}
	if m.Options.hasSWA() {
		t.Fatal("4.0-style config should not enable SWA")
	}
	if !m.Options.useRoPE(0) || !m.Options.useRoPE(3) {
		t.Fatal("non-SWA EXAONE should apply RoPE on all layers")
	}
}

func TestNewSWAConfig(t *testing.T) {
	cfg := baseConfig()
	cfg["exaone4.attention.sliding_window"] = uint32(4096)
	cfg["exaone4.attention.sliding_window_pattern"] = []bool{true, true, true, false}

	mm, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}

	m := mm.(*Model)
	if _, ok := m.Cache.(*kvcache.WrapperCache); !ok {
		t.Fatalf("cache = %T, want *kvcache.WrapperCache", m.Cache)
	}
	if !m.Options.isSWA(0) || m.Options.isSWA(3) {
		t.Fatalf("unexpected SWA pattern: %#v", m.Options.slidingWindowPattern)
	}
	if !m.Options.useRoPE(0) {
		t.Fatal("SWA layers should apply RoPE")
	}
	if m.Options.useRoPE(3) {
		t.Fatal("global layers in EXAONE SWA config should skip RoPE")
	}
}

func TestNewRejectsUnsupportedTokenizer(t *testing.T) {
	cfg := baseConfig()
	cfg["tokenizer.ggml.model"] = "llama"

	if _, err := New(cfg); err == nil {
		t.Fatal("expected unsupported tokenizer error")
	} else if err == model.ErrUnsupportedTokenizer {
		t.Fatal("expected contextual tokenizer error")
	}
}
