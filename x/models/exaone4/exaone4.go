// Package exaone4 provides an EXAONE 4 text decoder for MLX.
package exaone4

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

func init() {
	base.Register("Exaone4ForCausalLM", NewModel)
}

type RopeParameters struct {
	Type                          string  `json:"type"`
	RopeType                      string  `json:"rope_type"`
	Factor                        float32 `json:"factor"`
	LowFreqFactor                 float32 `json:"low_freq_factor"`
	HighFreqFactor                float32 `json:"high_freq_factor"`
	OriginalMaxPositionEmbeddings int32   `json:"original_max_position_embeddings"`
	RopeTheta                     float32 `json:"rope_theta"`
}

type Config struct {
	ModelType             string          `json:"model_type"`
	HiddenSize            int32           `json:"hidden_size"`
	IntermediateSize      int32           `json:"intermediate_size"`
	NumHiddenLayers       int32           `json:"num_hidden_layers"`
	NumAttentionHeads     int32           `json:"num_attention_heads"`
	NumKeyValueHeads      int32           `json:"num_key_value_heads"`
	VocabSize             int32           `json:"vocab_size"`
	MaxPositionEmbeddings int32           `json:"max_position_embeddings"`
	RMSNormEps            float32         `json:"rms_norm_eps"`
	RopeTheta             float32         `json:"rope_theta"`
	RopeParameters        *RopeParameters `json:"rope_parameters"`
	RopeScaling           *RopeParameters `json:"rope_scaling"`
	SlidingWindow         int32           `json:"sliding_window"`
	SlidingWindowPattern  string          `json:"sliding_window_pattern"`
	LayerTypes            []string        `json:"layer_types"`
	TieWordEmbeddings     bool            `json:"tie_word_embeddings"`

	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	HeadDim   int32      `json:"-"`
	Scale     float32    `json:"-"`
	RopeFreqs *mlx.Array `json:"-"`
}

type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*Layer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	tok *tokenizer.Tokenizer
	*Config

	weightPrefix string
	layouts      []TensorPathLayout
	discard      func(string) bool
}

type Layer struct {
	LayerIdx     int32
	IsSliding    bool
	Attention    *Attention
	MLP          *MLP
	PostAttnNorm *nn.RMSNorm
	PostFFNNorm  *nn.RMSNorm
}

type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer

	QNorm *nn.RMSNorm
	KNorm *nn.RMSNorm
}

type MLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
}

type TensorPathLayout struct {
	ContainerPrefix string
	ModelPrefix     string
}

func (l TensorPathLayout) modelPath(suffix string) string {
	return l.ContainerPrefix + l.ModelPrefix + suffix
}

func parseConfig(configData []byte) (Config, error) {
	var cfg Config
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config: %w", err)
	}
	return FinalizeConfig(cfg)
}

func FinalizeConfig(cfg Config) (Config, error) {
	if cfg.HiddenSize <= 0 {
		return Config{}, fmt.Errorf("invalid hidden_size: %d", cfg.HiddenSize)
	}
	if cfg.NumHiddenLayers <= 0 {
		return Config{}, fmt.Errorf("invalid num_hidden_layers: %d", cfg.NumHiddenLayers)
	}
	if cfg.NumAttentionHeads <= 0 {
		return Config{}, fmt.Errorf("invalid num_attention_heads: %d", cfg.NumAttentionHeads)
	}
	if cfg.NumKeyValueHeads <= 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
		return Config{}, fmt.Errorf("hidden_size (%d) must be divisible by num_attention_heads (%d)", cfg.HiddenSize, cfg.NumAttentionHeads)
	}
	if cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 {
		return Config{}, fmt.Errorf("num_attention_heads (%d) must be divisible by num_key_value_heads (%d)", cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}
	cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-5
	}
	if cfg.RopeParameters == nil {
		cfg.RopeParameters = cfg.RopeScaling
	}
	if cfg.RopeParameters != nil && cfg.RopeParameters.RopeTheta > 0 {
		cfg.RopeTheta = cfg.RopeParameters.RopeTheta
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 1000000
	}
	if cfg.MaxPositionEmbeddings <= 0 {
		cfg.MaxPositionEmbeddings = 8192
	}
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	cfg.RopeFreqs = buildLlama3RopeFreqs(int(cfg.HeadDim), cfg.RopeTheta, cfg.RopeParameters)
	return cfg, nil
}

func buildLlama3RopeFreqs(dim int, base float32, rp *RopeParameters) *mlx.Array {
	if rp == nil || dim <= 0 {
		return nil
	}
	ropeType := rp.RopeType
	if ropeType == "" {
		ropeType = rp.Type
	}
	if ropeType != "llama3" || rp.Factor <= 1 {
		return nil
	}
	lowFreqFactor := rp.LowFreqFactor
	if lowFreqFactor == 0 {
		lowFreqFactor = 1
	}
	highFreqFactor := rp.HighFreqFactor
	if highFreqFactor == 0 {
		highFreqFactor = 4
	}
	originalMax := rp.OriginalMaxPositionEmbeddings
	if originalMax <= 0 {
		originalMax = 8192
	}

	lowFreqWavelen := float64(originalMax) / float64(lowFreqFactor)
	highFreqWavelen := float64(originalMax) / float64(highFreqFactor)
	half := dim / 2
	freqs := make([]float32, half)
	for i := range half {
		posFreq := math.Pow(float64(base), float64(2*i)/float64(dim))
		invFreq := 1.0 / posFreq
		wavelen := 2 * math.Pi / invFreq
		scaledInvFreq := invFreq
		switch {
		case wavelen > lowFreqWavelen:
			scaledInvFreq = invFreq / float64(rp.Factor)
		case wavelen >= highFreqWavelen:
			smooth := (float64(originalMax)/wavelen - float64(lowFreqFactor)) / (float64(highFreqFactor) - float64(lowFreqFactor))
			scaledInvFreq = (1-smooth)*(invFreq/float64(rp.Factor)) + smooth*invFreq
		}
		freqs[i] = float32(1.0 / scaledInvFreq)
	}

	arr := mlx.FromValues(freqs, half)
	mlx.Eval(arr)
	return arr
}

func resolveTensorPathLayout(tensors map[string]*mlx.Array, layouts []TensorPathLayout) TensorPathLayout {
	if len(layouts) == 0 {
		layouts = []TensorPathLayout{{ModelPrefix: "model."}}
	}
	for _, layout := range layouts {
		if tensors[layout.modelPath("embed_tokens.weight")] != nil {
			return layout
		}
	}
	return layouts[0]
}

func isLayerSliding(layerIdx int32, cfg *Config) bool {
	if len(cfg.LayerTypes) == int(cfg.NumHiddenLayers) {
		return strings.EqualFold(cfg.LayerTypes[layerIdx], "sliding_attention")
	}
	if cfg.SlidingWindowPattern != "" {
		p := strings.ToUpper(cfg.SlidingWindowPattern)
		return p[layerIdx%int32(len(p))] == 'L'
	}
	return false
}

func NewModel(root *model.Root) (base.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	cfg, err := parseConfig(configData)
	if err != nil {
		return nil, err
	}

	return NewModelWithConfig(root, cfg)
}

type ModelOption func(*Model)

func WithTensorPathLayouts(layouts ...TensorPathLayout) ModelOption {
	return func(m *Model) {
		m.layouts = layouts
	}
}

func WithDiscardTensorFunc(discard func(string) bool) ModelOption {
	return func(m *Model) {
		m.discard = discard
	}
}

func NewModelWithConfig(root *model.Root, cfg Config, opts ...ModelOption) (base.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	if qt := root.QuantType(); qt != "" {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams(qt)
		if gs := root.GroupSize(); gs > 0 {
			cfg.QuantGroupSize = gs
		}
	} else {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams("")
	}
	cfg.TensorQuant = root.AllTensorQuant()

	tokData, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer config: %w", err)
	}
	tokConfig := &tokenizer.TokenizerConfig{ConfigJSON: configData}
	if genConfigData, err := root.Manifest.ReadConfig("generation_config.json"); err == nil {
		tokConfig.GenerationConfigJSON = genConfigData
	}
	if tokConfigData, err := root.Manifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokConfig.TokenizerConfigJSON = tokConfigData
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}

	m := &Model{
		Layers: make([]*Layer, cfg.NumHiddenLayers),
		Config: &cfg,
		tok:    tok,
		layouts: []TensorPathLayout{
			{ModelPrefix: "model."},
		},
	}
	for _, opt := range opts {
		opt(m)
	}
	for i := range m.Layers {
		m.Layers[i] = &Layer{LayerIdx: int32(i), IsSliding: isLayerSliding(int32(i), &cfg)}
	}
	return m, nil
}

func discardTensors(tensors map[string]*mlx.Array, discard func(string) bool) {
	if discard == nil {
		return
	}
	for name := range tensors {
		if discard(name) {
			delete(tensors, name)
		}
	}
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	discardTensors(tensors, m.discard)
	layout := resolveTensorPathLayout(tensors, m.layouts)
	m.weightPrefix = layout.ContainerPrefix
	modelPrefix := layout.ContainerPrefix + layout.ModelPrefix
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)

	embedTokens := model.MakeEmbeddingLayer(tensors, modelPrefix+"embed_tokens", m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	if embedTokens == nil {
		return fmt.Errorf("missing embedding weight: %sembed_tokens.weight", modelPrefix)
	}
	m.EmbedTokens = embedTokens

	normWeight := tensors[modelPrefix+"norm.weight"]
	if normWeight == nil {
		return fmt.Errorf("missing final norm weight: %snorm.weight", modelPrefix)
	}
	m.Norm = nn.NewRMSNorm(normWeight, m.RMSNormEps)

	if m.TieWordEmbeddings {
		m.LMHead = m.EmbedTokens.AsLinear()
	} else if lmHead := linears.Make(layout.ContainerPrefix + "lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else if lmHead := linears.Make("lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else {
		m.LMHead = m.EmbedTokens.AsLinear()
	}

	for i := int32(0); i < m.NumHiddenLayers; i++ {
		layerPrefix := fmt.Sprintf("%slayers.%d", modelPrefix, i)
		layer := &Layer{
			LayerIdx:  i,
			IsSliding: isLayerSliding(i, m.Config),
			Attention: &Attention{
				QProj: linears.Make(layerPrefix + ".self_attn.q_proj"),
				KProj: linears.Make(layerPrefix + ".self_attn.k_proj"),
				VProj: linears.Make(layerPrefix + ".self_attn.v_proj"),
				OProj: linears.Make(layerPrefix + ".self_attn.o_proj"),
			},
			MLP: &MLP{
				GateProj: linears.Make(layerPrefix + ".mlp.gate_proj"),
				UpProj:   linears.Make(layerPrefix + ".mlp.up_proj"),
				DownProj: linears.Make(layerPrefix + ".mlp.down_proj"),
			},
		}
		if w := tensors[layerPrefix+".self_attn.q_norm.weight"]; w != nil {
			layer.Attention.QNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".self_attn.k_norm.weight"]; w != nil {
			layer.Attention.KNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_attention_layernorm.weight"]; w != nil {
			layer.PostAttnNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_feedforward_layernorm.weight"]; w != nil {
			layer.PostFFNNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}

		if layer.Attention.QProj == nil || layer.Attention.KProj == nil || layer.Attention.VProj == nil || layer.Attention.OProj == nil {
			return fmt.Errorf("layer %d: missing attention projections", i)
		}
		if layer.Attention.QNorm == nil || layer.Attention.KNorm == nil {
			return fmt.Errorf("layer %d: missing attention q/k norms", i)
		}
		if layer.PostAttnNorm == nil || layer.PostFFNNorm == nil {
			return fmt.Errorf("layer %d: missing post layer norms", i)
		}
		if layer.MLP.GateProj == nil || layer.MLP.UpProj == nil || layer.MLP.DownProj == nil {
			return fmt.Errorf("layer %d: missing mlp projections", i)
		}
		m.Layers[i] = layer
	}

	return nil
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))

	h := m.TokenEmbeddings(b.InputIDs)
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, b, c, positions, B, L, m.Config)
	}

	return m.Norm.Forward(h, m.RMSNormEps)
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	return m.LMHead.Forward(x)
}

func (m *Model) TokenEmbeddings(inputIDs *mlx.Array) *mlx.Array {
	return m.EmbedTokens.Forward(inputIDs)
}

func (m *Model) NumLayers() int {
	return len(m.Layers)
}

func (m *Model) MaxContextLength() int {
	return int(m.MaxPositionEmbeddings)
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer {
	return m.tok
}

func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i, layer := range m.Layers {
		if m.SlidingWindow > 0 && layer.IsSliding {
			caches[i] = cache.NewRotatingKVCache(max(1, int(m.SlidingWindow)-1))
		} else {
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	attnOut := l.Attention.Forward(x, b, c, positions, B, L, l.IsSliding, cfg)
	h := mlx.Add(x, l.PostAttnNorm.Forward(attnOut, cfg.RMSNormEps))
	ffnOut := l.MLP.Forward(h)
	return mlx.Add(h, l.PostFFNNorm.Forward(ffnOut, cfg.RMSNormEps))
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, isSliding bool, cfg *Config) *mlx.Array {
	q := a.QProj.Forward(x)
	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)
	q = a.QNorm.Forward(q, cfg.RMSNormEps)

	k := a.KProj.Forward(x)
	k = mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	k = a.KNorm.Forward(k, cfg.RMSNormEps)

	v := a.VProj.Forward(x)
	v = mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	if isSliding || cfg.SlidingWindow <= 0 {
		q = mlx.RoPEWithFreqs(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions, cfg.RopeFreqs)
		k = mlx.RoPEWithFreqs(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions, cfg.RopeFreqs)
	}

	var kv nn.SDPAOption
	mask := nn.CausalMask()
	if c != nil {
		history := c.(cache.Attention).Update(b, k, v)
		kv = nn.WithKVHistory(history)
	} else {
		kv = nn.WithKV(k, v, b.SeqQueryLens)
		if isSliding && cfg.SlidingWindow > 0 {
			mask = mask.Intersect(nn.SlidingWindowMask(b, k.Dim(2), int(cfg.SlidingWindow), q.DType()))
		}
	}

	out := nn.ScaledDotProductAttention(b, q, cfg.Scale, kv, nn.WithMask(mask))
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	return m.DownProj.Forward(mlx.SwiGLU(m.GateProj.Forward(x), m.UpProj.Forward(x)))
}
