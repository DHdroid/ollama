package exaone4_5

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/exaone4"
	"github.com/ollama/ollama/x/models/nn"
)

var (
	_ base.DraftModel          = (*MTPModel)(nil)
	_ base.MTPDraftModel       = (*MTPModel)(nil)
	_ base.CachedMTPDraftModel = (*MTPModel)(nil)
)

type MTPModel struct {
	FC                 nn.LinearLayer
	PreFCNormEmbedding *nn.RMSNorm
	PreFCNormHidden    *nn.RMSNorm
	Layer              *exaone4.Layer
	Norm               *nn.RMSNorm

	target *Model
	*exaone4.Config
}

func NewMTPModel(root *model.Root, target base.Model) (base.DraftModel, error) {
	targetModel, ok := target.(*Model)
	if !ok {
		return nil, fmt.Errorf("EXAONE 4.5 MTP requires EXAONE 4.5 target, got %T", target)
	}

	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	cfg, err := parseTextConfig(configData)
	if err != nil {
		return nil, err
	}
	cfg.NumHiddenLayers = 1
	cfg.LayerTypes = nil
	cfg.SlidingWindowPattern = ""
	cfg.SlidingWindow = 0

	if qt := root.QuantType(); qt != "" {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams(qt)
		if gs := root.GroupSize(); gs > 0 {
			cfg.QuantGroupSize = gs
		}
	} else {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams("")
	}
	cfg.TensorQuant = root.AllTensorQuant()

	return &MTPModel{
		target: targetModel,
		Config: &cfg,
		Layer: &exaone4.Layer{
			LayerIdx:  0,
			Attention: &exaone4.Attention{},
			MLP:       &exaone4.MLP{},
		},
	}, nil
}

func (m *MTPModel) LoadWeights(tensors map[string]*mlx.Array) error {
	prefix := "mtp."
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)

	m.FC = linears.Make(prefix + "fc")
	if m.FC == nil {
		return fmt.Errorf("missing EXAONE 4.5 MTP fc weight")
	}
	if w := tensors[prefix+"pre_fc_norm_embedding.weight"]; w != nil {
		m.PreFCNormEmbedding = nn.NewRMSNorm(w, m.RMSNormEps)
	} else {
		return fmt.Errorf("missing EXAONE 4.5 MTP pre_fc_norm_embedding weight")
	}
	if w := tensors[prefix+"pre_fc_norm_hidden.weight"]; w != nil {
		m.PreFCNormHidden = nn.NewRMSNorm(w, m.RMSNormEps)
	} else {
		return fmt.Errorf("missing EXAONE 4.5 MTP pre_fc_norm_hidden weight")
	}
	if w := tensors[prefix+"norm.weight"]; w != nil {
		m.Norm = nn.NewRMSNorm(w, m.RMSNormEps)
	} else {
		return fmt.Errorf("missing EXAONE 4.5 MTP final norm weight")
	}

	layerPrefix := prefix + "layers.0"
	m.Layer.Attention = &exaone4.Attention{
		QProj: linears.Make(layerPrefix + ".self_attn.q_proj"),
		KProj: linears.Make(layerPrefix + ".self_attn.k_proj"),
		VProj: linears.Make(layerPrefix + ".self_attn.v_proj"),
		OProj: linears.Make(layerPrefix + ".self_attn.o_proj"),
	}
	m.Layer.MLP = &exaone4.MLP{
		GateProj: linears.Make(layerPrefix + ".mlp.gate_proj"),
		UpProj:   linears.Make(layerPrefix + ".mlp.up_proj"),
		DownProj: linears.Make(layerPrefix + ".mlp.down_proj"),
	}
	if w := tensors[layerPrefix+".self_attn.q_norm.weight"]; w != nil {
		m.Layer.Attention.QNorm = nn.NewRMSNorm(w, m.RMSNormEps)
	}
	if w := tensors[layerPrefix+".self_attn.k_norm.weight"]; w != nil {
		m.Layer.Attention.KNorm = nn.NewRMSNorm(w, m.RMSNormEps)
	}
	if w := tensors[layerPrefix+".post_attention_layernorm.weight"]; w != nil {
		m.Layer.PostAttnNorm = nn.NewRMSNorm(w, m.RMSNormEps)
	}
	if w := tensors[layerPrefix+".post_feedforward_layernorm.weight"]; w != nil {
		m.Layer.PostFFNNorm = nn.NewRMSNorm(w, m.RMSNormEps)
	}

	if m.Layer.Attention.QProj == nil || m.Layer.Attention.KProj == nil || m.Layer.Attention.VProj == nil || m.Layer.Attention.OProj == nil {
		return fmt.Errorf("missing EXAONE 4.5 MTP attention projections")
	}
	if m.Layer.Attention.QNorm == nil || m.Layer.Attention.KNorm == nil {
		return fmt.Errorf("missing EXAONE 4.5 MTP attention q/k norms")
	}
	if m.Layer.PostAttnNorm == nil || m.Layer.PostFFNNorm == nil {
		return fmt.Errorf("missing EXAONE 4.5 MTP post layer norms")
	}
	if m.Layer.MLP.GateProj == nil || m.Layer.MLP.UpProj == nil || m.Layer.MLP.DownProj == nil {
		return fmt.Errorf("missing EXAONE 4.5 MTP mlp projections")
	}

	return nil
}

func (m *MTPModel) NewCaches() []cache.Cache {
	return []cache.Cache{cache.NewKVCache()}
}

func (m *MTPModel) AppendContext(target base.MTPEmbeddingModel, inputIDs, hidden *mlx.Array, position int32, caches []cache.Cache) {
	tokenEmbedding := target.TokenEmbeddings(inputIDs)
	inputs := tokenEmbedding.Concatenate(-1, hidden)
	m.forward(inputs, position, caches)
}

func (m *MTPModel) Draft(inputs *mlx.Array, position int32, caches []cache.Cache) (logits, hidden *mlx.Array) {
	hidden = m.forward(inputs, position, caches)
	return m.target.Unembed(hidden), hidden
}

func (m *MTPModel) forward(inputs *mlx.Array, position int32, caches []cache.Cache) *mlx.Array {
	dims := inputs.Dims()
	if len(dims) != 3 || dims[2] != int(m.HiddenSize*2) {
		panic(fmt.Sprintf("EXAONE 4.5 MTP input shape = %v, want [B L %d]", dims, m.HiddenSize*2))
	}

	B, L := int32(dims[0]), int32(dims[1])
	embedding := inputs.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(m.HiddenSize)))
	targetHidden := inputs.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(int(m.HiddenSize), mlx.End))
	embedding = m.PreFCNormEmbedding.Forward(embedding, m.RMSNormEps)
	targetHidden = m.PreFCNormHidden.Forward(targetHidden, m.RMSNormEps)
	h := m.FC.Forward(embedding.Concatenate(-1, targetHidden))

	b := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, int(B), int(L)),
		SeqOffsets:   []int32{position},
		SeqQueryLens: []int32{L},
	}
	positions := mlx.FromValues([]int32{position}, 1)
	var c cache.Cache
	if len(caches) > 0 {
		c = caches[0]
	}
	h = m.Layer.Forward(h, b, c, positions, B, L, m.Config)
	return m.Norm.Forward(h, m.RMSNormEps)
}
