package exaone4_5

import (
	"iter"
	"testing"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/model/models/exaone4"
)

type testConfig map[string]any

func (c testConfig) Architecture() string { return "exaone4_5" }

func (c testConfig) key(key string) string {
	switch {
	case len(key) >= len("tokenizer.") && key[:len("tokenizer.")] == "tokenizer.":
		return key
	case len(key) >= len("general.") && key[:len("general.")] == "general.":
		return key
	default:
		return "exaone4_5." + key
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

func TestNewWrapperOwnsLanguageModel(t *testing.T) {
	cfg := testConfig{
		"general.architecture":                       "exaone4_5",
		"general.basename":                           "EXAONE-4.5",
		"tokenizer.ggml.model":                       "gpt2",
		"tokenizer.ggml.pre":                         "exaone-moe",
		"tokenizer.ggml.tokens":                      []string{"[PAD]", "[BOS]", "[EOS]", "a"},
		"tokenizer.ggml.token_type":                  []int32{3, 3, 3, 1},
		"tokenizer.ggml.merges":                      []string{},
		"tokenizer.ggml.bos_token_id":                uint32(1),
		"tokenizer.ggml.eos_token_id":                uint32(2),
		"exaone4_5.block_count":                      uint32(4),
		"exaone4_5.embedding_length":                 uint32(128),
		"exaone4_5.attention.head_count":             uint32(8),
		"exaone4_5.attention.head_count_kv":          uint32(2),
		"exaone4_5.attention.key_length":             uint32(16),
		"exaone4_5.attention.value_length":           uint32(16),
		"exaone4_5.attention.layer_norm_rms_epsilon": float32(1e-5),
		"exaone4_5.attention.sliding_window":         uint32(4096),
		"exaone4_5.attention.sliding_window_pattern": []bool{true, true, true, false},
		"exaone4_5.rope.freq_base":                   float32(1000000),
	}

	mm, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}

	m := mm.(*Model)
	if _, ok := any(m.LanguageModel).(*exaone4.Model); !ok {
		t.Fatalf("language model = %T, want *exaone4.Model", m.LanguageModel)
	}
	if _, ok := m.Cache.(*kvcache.WrapperCache); !ok {
		t.Fatalf("cache = %T, want *kvcache.WrapperCache", m.Cache)
	}
	if !m.LanguageModel.Options.UseRoPE(0) {
		t.Fatal("SWA layer should apply RoPE")
	}
	if m.LanguageModel.Options.UseRoPE(3) {
		t.Fatal("global layer should skip RoPE")
	}
}
