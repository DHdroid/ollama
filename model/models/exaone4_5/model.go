package exaone4_5

import (
	"bytes"
	"fmt"
	"image"
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
	*VisionModel  `gguf:"v,alt:visual"`
	ImageProcessor

	imageToken  int32
	visionStart int32
	visionEnd   int32
}

var _ model.Model = (*Model)(nil)
var _ model.MultimodalProcessor = (*Model)(nil)

func New(c fs.Config) (model.Model, error) {
	if c.String("tokenizer.ggml.model") != "gpt2" {
		return nil, fmt.Errorf("unsupported tokenizer: %s", c.String("tokenizer.ggml.model"))
	}

	t := exaone45Tokenizer(c)
	languageModel := exaone4.NewTextModel(c, t, exaone45UseRoPE(c))
	exaone4.ConfigureCache(languageModel, c)

	m := &Model{
		Tokenizer:      t,
		LanguageModel:  languageModel,
		VisionModel:    newVisionModel(c),
		ImageProcessor: newImageProcessor(c),
		imageToken:     tokenID(c, "<|image_pad|>", 67),
		visionStart:    tokenID(c, "<vision>", 73),
		visionEnd:      tokenID(c, "</vision>", 74),
	}
	m.Cache = languageModel.Cache
	return m, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	if len(batch.Multimodal) == 0 {
		return m.LanguageModel.Forward(ctx, batch)
	}

	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))
	hiddenStates := m.LanguageModel.TokenEmbedding.Forward(ctx, batch.Inputs).Duplicate(ctx)

	for _, mi := range batch.Multimodal {
		img := mi.Multimodal[0].Tensor
		ctx.Forward(img.Copy(ctx, hiddenStates.View(ctx, mi.Index*hiddenStates.Stride(1), img.Dim(0)*img.Dim(1))))
	}

	return m.LanguageModel.ForwardWithHiddenStates(ctx, batch, positions, hiddenStates)
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	if m.VisionModel == nil || len(m.VisionModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	img, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	f32s, grid, err := m.ImageProcessor.ProcessImage(img)
	if err != nil {
		return nil, err
	}

	patchDim := m.numChannels * m.temporalPatchSize * m.patchSize * m.patchSize
	numPatches := grid.Temporal * grid.Height * grid.Width
	pixelValues := ctx.Input().FromFloats(f32s, patchDim, numPatches)

	visionOutputs := m.VisionModel.Forward(ctx, pixelValues, grid)
	return []input.Multimodal{{Tensor: visionOutputs, Data: grid}}, nil
}

func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	var result []*input.Input

	for _, inp := range inputs {
		if len(inp.Multimodal) == 0 {
			result = append(result, inp)
			continue
		}

		tokensPerGrid := inp.Multimodal[0].Tensor.Dim(1)
		result = append(result, &input.Input{Token: m.visionStart})
		result = append(result, &input.Input{
			Token:          m.imageToken,
			Multimodal:     inp.Multimodal,
			MultimodalHash: inp.MultimodalHash,
			SameBatch:      tokensPerGrid,
		})

		for range tokensPerGrid - 1 {
			result = append(result, &input.Input{Token: m.imageToken})
		}

		result = append(result, &input.Input{Token: m.visionEnd})
	}

	return result, nil
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

func tokenID(c fs.Config, token string, fallback int32) int32 {
	for i, value := range c.Strings("tokenizer.ggml.tokens") {
		if value == token {
			return int32(i)
		}
	}

	return fallback
}
