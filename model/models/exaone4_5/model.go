package exaone4_5

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/model/models/exaone4"
	"github.com/ollama/ollama/tokenizer"
)

const exaoneMoePretokenizer = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]?|\s*[\r\n]|\s+(?!\S)|\s+`

type Model struct {
	model.Base
	tokenizer.Tokenizer

	LanguageModel *exaone4.Model
	VisionModel   any
}

var _ model.Model = (*Model)(nil)

func New(c fs.Config) (model.Model, error) {
	if c.String("tokenizer.ggml.model") != "gpt2" {
		return nil, fmt.Errorf("unsupported tokenizer: %s", c.String("tokenizer.ggml.model"))
	}

	t := exaone45Tokenizer(c)
	languageModel := exaone4.NewTextModel(c, t, exaone45UseRoPE(c))
	exaone4.ConfigureCache(languageModel, c)

	m := &Model{
		Tokenizer:     t,
		LanguageModel: languageModel,
	}
	m.Cache = languageModel.Cache
	return m, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	return m.LanguageModel.Forward(ctx, batch)
}

func exaone45UseRoPE(c fs.Config) exaone4.RopePolicy {
	pattern := c.Bools("attention.sliding_window_pattern")
	if len(pattern) == 0 {
		return nil
	}

	return func(layer int) bool {
		return layer < len(pattern) && pattern[layer]
	}
}

func exaone45Tokenizer(c fs.Config) tokenizer.Tokenizer {
	vocabulary := tokenizer.Vocabulary{
		Values: c.Strings("tokenizer.ggml.tokens"),
		Types:  c.Ints("tokenizer.ggml.token_type"),
		Merges: c.Strings("tokenizer.ggml.merges"),
		AddBOS: c.Bool("tokenizer.ggml.add_bos_token", false),
		BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
		AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
		EOS: append(
			[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
			c.Ints("tokenizer.ggml.eos_token_ids")...,
		),
	}

	if strings.EqualFold(c.String("tokenizer.ggml.pre"), "exaone-moe") {
		return tokenizer.NewBytePairEncoding(
			&vocabulary,
			exaoneMoePretokenizer,
		)
	}

	return tokenizer.NewBytePairEncoding(&vocabulary)
}
