// Package exaone4_5 registers EXAONE 4.5 architectures for MLX.
package exaone4_5

import (
	"encoding/json"
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/exaone4"
	"github.com/ollama/ollama/x/tokenizer"
)

func init() {
	base.Register("Exaone4_5_ForConditionalGeneration", NewModel)
	base.RegisterDraft("Exaone4_5_ForConditionalGenerationMTP", NewMTPModel)
}

type Model struct {
	LanguageModel *exaone4.Model
	VisionModel   any
}

func NewModel(root *model.Root) (base.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	cfg, err := parseTextConfig(configData)
	if err != nil {
		return nil, err
	}

	languageModel, err := exaone4.NewModelWithConfig(root, cfg,
		exaone4.WithTensorPathLayouts(
			exaone4.TensorPathLayout{ContainerPrefix: "model.language_model."},
			exaone4.TensorPathLayout{ContainerPrefix: "model.language_model.", ModelPrefix: "model."},
			exaone4.TensorPathLayout{ContainerPrefix: "language_model."},
			exaone4.TensorPathLayout{ContainerPrefix: "language_model.", ModelPrefix: "model."},
			exaone4.TensorPathLayout{ModelPrefix: "model."},
		),
	)
	if err != nil {
		return nil, err
	}

	lm, ok := languageModel.(*exaone4.Model)
	if !ok {
		return nil, fmt.Errorf("unexpected EXAONE 4.5 language model type: %T", languageModel)
	}

	return &Model{LanguageModel: lm}, nil
}

func parseTextConfig(configData []byte) (exaone4.Config, error) {
	var envelope struct {
		TextConfig        json.RawMessage `json:"text_config"`
		TieWordEmbeddings *bool           `json:"tie_word_embeddings"`
		VocabSize         int32           `json:"vocab_size"`
	}
	if err := json.Unmarshal(configData, &envelope); err != nil {
		return exaone4.Config{}, fmt.Errorf("parse config envelope: %w", err)
	}
	if len(envelope.TextConfig) == 0 {
		return exaone4.Config{}, fmt.Errorf("missing text_config")
	}

	var cfg exaone4.Config
	if err := json.Unmarshal(envelope.TextConfig, &cfg); err != nil {
		return exaone4.Config{}, fmt.Errorf("parse text_config: %w", err)
	}
	if envelope.TieWordEmbeddings != nil {
		cfg.TieWordEmbeddings = *envelope.TieWordEmbeddings
	}
	if cfg.VocabSize == 0 && envelope.VocabSize > 0 {
		cfg.VocabSize = envelope.VocabSize
	}
	return exaone4.FinalizeConfig(cfg)
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	return m.LanguageModel.LoadWeights(tensors)
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array {
	return m.LanguageModel.Forward(b, caches)
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	return m.LanguageModel.Unembed(x)
}

func (m *Model) TokenEmbeddings(inputIDs *mlx.Array) *mlx.Array {
	return m.LanguageModel.TokenEmbeddings(inputIDs)
}

func (m *Model) NumLayers() int {
	return m.LanguageModel.NumLayers()
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer {
	return m.LanguageModel.Tokenizer()
}

func (m *Model) MaxContextLength() int {
	return m.LanguageModel.MaxContextLength()
}

func (m *Model) MTPDraftDefaults(bool) base.MTPDefaults {
	return base.MTPDefaults{
		InitialDraftTokens: 3,
		MaxDraftTokens:     3,
		Enabled:            true,
	}
}
