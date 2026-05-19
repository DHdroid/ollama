package exaone4

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/tokenizer"
)

const (
	cacheTypeSWA = iota
	cacheTypeCausal

	cacheRecurrentSeqLen = 4096
)

type RopePolicy func(layer int) bool

type Options struct {
	hiddenSize,
	numHeads,
	numKVHeads,
	headSize int

	eps,
	ropeBase float32

	slidingWindowPattern []bool
	UseRoPE              RopePolicy
}

func (o Options) headDim() int {
	return o.headSize
}

func (o Options) isSWA(layer int) bool {
	return layer < len(o.slidingWindowPattern) && o.slidingWindowPattern[layer]
}

func (o Options) hasSWA() bool {
	return len(o.slidingWindowPattern) > 0
}

func (o Options) useRoPE(layer int) bool {
	if o.UseRoPE != nil {
		return o.UseRoPE(layer)
	}
	return true
}

func (o Options) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions, factors ml.Tensor) ml.Tensor {
	opts := []func(*rope.Options){rope.WithTypeNeoX(), rope.WithFactors(factors)}
	return nn.RoPE(ctx, states, positions, o.headDim(), o.ropeBase, 1., opts...)
}

type Attention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Output    *nn.Linear  `gguf:"attn_output"`
}

func (sa *Attention) Forward(ctx ml.Context, layer int, hiddenStates, positions, ropeFactors ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	batchSize := hiddenStates.Dim(1)
	headDim := opts.headDim()

	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, headDim, opts.numHeads, batchSize)
	key = key.Reshape(ctx, headDim, opts.numKVHeads, batchSize)
	value = value.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	query = sa.QueryNorm.Forward(ctx, query, opts.eps)
	key = sa.KeyNorm.Forward(ctx, key, opts.eps)

	if opts.useRoPE(layer) {
		query = opts.applyRotaryPositionEmbeddings(ctx, query, positions, ropeFactors)
		key = opts.applyRotaryPositionEmbeddings(ctx, key, positions, ropeFactors)
	}

	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(headDim)), cache)
	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), batchSize)
	return sa.Output.Forward(ctx, attention)
}

type MLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenStates ml.Tensor) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).
		SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

type Layer struct {
	*Attention
	PostAttentionNorm *nn.RMSNorm `gguf:"post_attention_norm"`
	*MLP
	PostFFWNorm *nn.RMSNorm `gguf:"post_ffw_norm"`
}

func (l *Layer) Forward(ctx ml.Context, layer int, hiddenStates, positions, outputs, ropeFactors ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	residual := hiddenStates
	hiddenStates = l.Attention.Forward(ctx, layer, hiddenStates, positions, ropeFactors, cache, opts)

	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = l.PostAttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = l.MLP.Forward(ctx, hiddenStates)
	hiddenStates = l.PostFFWNorm.Forward(ctx, hiddenStates, opts.eps)
	return hiddenStates.Add(ctx, residual)
}

type Model struct {
	model.Base
	tokenizer.Tokenizer

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`
	RopeFactors    ml.Tensor     `gguf:"rope_freqs.weight"`

	Layers []Layer `gguf:"blk"`

	*Options
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))
	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	for i, layer := range m.Layers {
		if m.Cache != nil {
			m.Cache.SetLayer(i)
			if m.Options.hasSWA() {
				cacheType := cacheTypeCausal
				if m.Options.isSWA(i) {
					cacheType = cacheTypeSWA
				}
				m.Cache.(*kvcache.WrapperCache).SetLayerType(cacheType)
			}
		}

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, i, hiddenStates, positions, outputs, m.RopeFactors, m.Cache, m.Options)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	if !m.Options.useRoPE(layer) {
		return key, nil
	}

	return m.Options.applyRotaryPositionEmbeddings(ctx, key, shift, m.RopeFactors), nil
}

var _ model.Model = (*Model)(nil)

func New(c fs.Config) (model.Model, error) {
	if c.String("tokenizer.ggml.model") != "gpt2" {
		return nil, fmt.Errorf("unsupported tokenizer: %s", c.String("tokenizer.ggml.model"))
	}

	m := NewTextModel(c, exaone4Tokenizer(c), nil)
	ConfigureCache(m, c)

	return m, nil
}

func NewTextModel(c fs.Config, t tokenizer.Tokenizer, useRoPE RopePolicy) *Model {
	hiddenSize := int(c.Uint("embedding_length"))
	numHeads := int(c.Uint("attention.head_count"))
	headDim := int(c.Uint("attention.key_length", uint32(hiddenSize/numHeads)))

	slidingWindowPattern := c.Bools("attention.sliding_window_pattern")
	if useRoPE == nil && len(slidingWindowPattern) > 0 {
		useRoPE = func(layer int) bool {
			return layer < len(slidingWindowPattern) && slidingWindowPattern[layer]
		}
	}

	return &Model{
		Tokenizer: t,
		Layers:    make([]Layer, c.Uint("block_count")),
		Options: &Options{
			hiddenSize:           hiddenSize,
			numHeads:             numHeads,
			numKVHeads:           int(c.Uint("attention.head_count_kv")),
			headSize:             headDim,
			eps:                  c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:             c.Float("rope.freq_base"),
			slidingWindowPattern: slidingWindowPattern,
			UseRoPE:              useRoPE,
		},
	}
}

func ConfigureCache(m *Model, c fs.Config) {
	if m.Options.hasSWA() {
		slidingWindowLen := int32(c.Uint("attention.sliding_window"))
		m.Cache = kvcache.NewWrapperCache(
			kvcache.NewSWAMemCache(slidingWindowLen, cacheRecurrentSeqLen, m.Shift),
			kvcache.NewCausalCache(m.Shift),
		)
	} else {
		m.Cache = kvcache.NewCausalCache(m.Shift)
	}
}

func exaone4Tokenizer(c fs.Config) tokenizer.Tokenizer {
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

	return tokenizer.NewBytePairEncoding(&vocabulary)
}
