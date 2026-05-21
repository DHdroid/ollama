package exaone4_5

import (
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
)

func blockDiagonalMask(ctx ml.Context, seqLength int, bounds []int) ml.Tensor {
	s := make([][]float32, seqLength)
	for i := range s {
		s[i] = slices.Repeat([]float32{float32(math.Inf(-1))}, seqLength)
	}

	for i := 1; i < len(bounds); i++ {
		start, end := bounds[i-1], bounds[i]
		for row := start; row < end; row++ {
			for col := start; col < end; col++ {
				s[row][col] = 0
			}
		}
	}

	return ctx.Input().FromFloats(slices.Concat(s...), seqLength, seqLength)
}

type VisionSelfAttention struct {
	QKV    *nn.Linear `gguf:"attn_qkv"`
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_out"`
}

func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenStates, positions, mask ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	batchSize := hiddenStates.Dim(1)

	var query, key, value ml.Tensor
	if sa.QKV != nil {
		qkv := sa.QKV.Forward(ctx, hiddenStates)
		qkv = qkv.Reshape(ctx, opts.headDim, -1, batchSize)
		chunks := qkv.ChunkSections(ctx, 1, opts.numHeads, opts.numKVHeads, opts.numKVHeads)
		query, key, value = chunks[0], chunks[1], chunks[2]
	} else {
		query = sa.Query.Forward(ctx, hiddenStates)
		key = sa.Key.Forward(ctx, hiddenStates)
		value = sa.Value.Forward(ctx, hiddenStates)

		query = query.Reshape(ctx, opts.headDim, opts.numHeads, batchSize)
		key = key.Reshape(ctx, opts.headDim, opts.numKVHeads, batchSize)
		value = value.Reshape(ctx, opts.headDim, opts.numKVHeads, batchSize)
	}

	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions)
	key = opts.applyRotaryPositionEmbeddings(ctx, key, positions)

	scale := 1.0 / math.Sqrt(float64(opts.headDim))
	if sdpa, ok := query.(ml.ScaledDotProductAttention); ok {
		attention := sdpa.ScaledDotProductAttention(ctx, key, value, mask, nil, nil, scale, false)
		attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2))
		return sa.Output.Forward(ctx, attention)
	}

	if opts.numKVHeads != opts.numHeads {
		key = repeatVisionKVHeads(ctx, key, opts)
		value = repeatVisionKVHeads(ctx, value, opts)
	}

	query = query.Permute(ctx, 0, 2, 1, 3)
	key = key.Permute(ctx, 0, 2, 1, 3)
	value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	kq := key.MulmatFullPrec(ctx, query)
	kq = kq.Scale(ctx, scale)
	if mask != nil {
		kq = kq.Add(ctx, mask)
	}
	kq = kq.Softmax(ctx)

	kqv := value.Mulmat(ctx, kq)
	attention := kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2))
	return sa.Output.Forward(ctx, attention)
}

func repeatVisionKVHeads(ctx ml.Context, states ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	if opts.numKVHeads == 0 || opts.numHeads%opts.numKVHeads != 0 {
		panic("exaone4_5 vision attention requires numHeads to be divisible by numKVHeads")
	}

	repeatFactor := opts.numHeads / opts.numKVHeads
	seqLength := states.Dim(2)
	states = states.Reshape(ctx, opts.headDim, 1, opts.numKVHeads*seqLength)
	states = states.Repeat4D(ctx, opts.headDim, repeatFactor, opts.numKVHeads*seqLength, 1)
	return states.Reshape(ctx, opts.headDim, opts.numHeads, seqLength)
}

type VisionMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

type VisionEncoderLayer struct {
	Norm1         *nn.RMSNorm `gguf:"ln1"`
	SelfAttention *VisionSelfAttention
	Norm2         *nn.RMSNorm `gguf:"ln2"`
	MLP           *VisionMLP
}

func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenStates, positions, mask ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	residual := hiddenStates
	hiddenStates = e.Norm1.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.SelfAttention.Forward(ctx, hiddenStates, positions, mask, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = e.Norm2.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.MLP.Forward(ctx, hiddenStates)
	return hiddenStates.Add(ctx, residual)
}

type VisionModelOptions struct {
	hiddenSize,
	numHeads,
	numKVHeads,
	headDim,
	patchSize,
	numChannels,
	spatialMergeSize,
	windowSize,
	temporalPatchSize int

	eps,
	ropeTheta float32

	fullAttnBlocks []int32
}

func (o VisionModelOptions) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	section := o.headDim / 4
	return nn.RoPE(ctx, states, positions, o.headDim/2, o.ropeTheta, 1,
		rope.WithVision([]int{section, section, 0, 0}),
	)
}

type PatchEmbedding struct {
	PatchConv0 *nn.Conv2D `gguf:"patch_embd_0"`
	PatchConv1 *nn.Conv2D `gguf:"patch_embd_1"`
}

func (pe *PatchEmbedding) Forward(ctx ml.Context, pixelValues ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	numPatches := pixelValues.Shape()[1]

	pixelValues = pixelValues.Reshape(ctx, opts.patchSize*opts.patchSize, opts.temporalPatchSize, opts.numChannels, numPatches)
	pixelValues = pixelValues.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	in0 := pixelValues.View(ctx, 0, 1, pixelValues.Stride(1), pixelValues.Dim(1), pixelValues.Stride(2), pixelValues.Dim(2), pixelValues.Stride(3), pixelValues.Dim(3)).Contiguous(ctx)
	in0 = in0.Reshape(ctx, opts.patchSize, opts.patchSize, opts.numChannels, numPatches)

	hiddenStates := pe.PatchConv0.Forward(ctx, in0, opts.patchSize, opts.patchSize, 0, 0, 1, 1)
	if pe.PatchConv1 != nil && opts.temporalPatchSize > 1 {
		in1 := pixelValues.View(ctx, pixelValues.Stride(0), 1, pixelValues.Stride(1), pixelValues.Dim(1), pixelValues.Stride(2), pixelValues.Dim(2), pixelValues.Stride(3), pixelValues.Dim(3)).Contiguous(ctx)
		in1 = in1.Reshape(ctx, opts.patchSize, opts.patchSize, opts.numChannels, numPatches)
		hiddenStates = hiddenStates.Add(ctx, pe.PatchConv1.Forward(ctx, in1, opts.patchSize, opts.patchSize, 0, 0, 1, 1))
	}

	return hiddenStates.Reshape(ctx, opts.hiddenSize, numPatches)
}

type VisionPatchMerger struct {
	LNQ  *nn.RMSNorm `gguf:"ln_q"`
	MLP0 *nn.Linear  `gguf:"mlp.0"`
	MLP2 *nn.Linear  `gguf:"mlp.2"`
}

func (pm *VisionPatchMerger) Forward(ctx ml.Context, visionOutputs ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	normalized := pm.LNQ.Forward(ctx, visionOutputs, opts.eps)
	spatialMergeUnit := opts.spatialMergeSize * opts.spatialMergeSize
	hiddenSize := visionOutputs.Dim(0) * spatialMergeUnit
	reshaped := normalized.Reshape(ctx, hiddenSize, normalized.Dim(1)/spatialMergeUnit)
	return pm.MLP2.Forward(ctx, pm.MLP0.Forward(ctx, reshaped).GELU(ctx))
}

type VisionModel struct {
	PatchEmbedding *PatchEmbedding
	Layers         []VisionEncoderLayer `gguf:"blk"`
	PatchMerger    *VisionPatchMerger   `gguf:"merger"`

	*VisionModelOptions
}

func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor, grid *Grid) ml.Tensor {
	hiddenStates := m.PatchEmbedding.Forward(ctx, pixelValues, m.VisionModelOptions)

	index, bounds := m.windowIndex(grid)
	spatialMergeUnit := m.spatialMergeSize * m.spatialMergeSize

	windowIndex := ctx.Input().FromInts(index, len(index))
	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0)*spatialMergeUnit, hiddenStates.Dim(1)/spatialMergeUnit)
	hiddenStates = hiddenStates.Rows(ctx, windowIndex.Argsort(ctx))
	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0)/spatialMergeUnit, hiddenStates.Dim(1)*spatialMergeUnit)

	positions := ctx.Input().FromInts(func() []int32 {
		zero := make([]int32, grid.Height*grid.Width)
		s := [][]int32{
			make([]int32, grid.Height*grid.Width),
			make([]int32, grid.Height*grid.Width),
			zero,
			zero,
		}

		var cur int
		for y := 0; y < grid.Height; y += m.spatialMergeSize {
			for x := 0; x < grid.Width; x += m.spatialMergeSize {
				for dy := range m.spatialMergeSize {
					for dx := range m.spatialMergeSize {
						i := int(index[cur/spatialMergeUnit]) * spatialMergeUnit
						i += cur % spatialMergeUnit
						s[0][i] = int32(y + dy)
						s[1][i] = int32(x + dx)
						cur++
					}
				}
			}
		}

		return slices.Concat(s...)
	}(), grid.Height*grid.Width*4)

	mask := blockDiagonalMask(ctx, hiddenStates.Dim(1), bounds)
	for i, layer := range m.Layers {
		if slices.Contains(m.fullAttnBlocks, int32(i)) {
			hiddenStates = layer.Forward(ctx, hiddenStates, positions, nil, m.VisionModelOptions)
		} else {
			hiddenStates = layer.Forward(ctx, hiddenStates, positions, mask, m.VisionModelOptions)
		}
	}

	hiddenStates = m.PatchMerger.Forward(ctx, hiddenStates, m.VisionModelOptions)
	return hiddenStates.Rows(ctx, windowIndex)
}

func (m *VisionModel) windowIndex(grid *Grid) (index []int32, bounds []int) {
	height := grid.Height / m.spatialMergeSize
	width := grid.Width / m.spatialMergeSize
	window := m.windowSize / m.patchSize / m.spatialMergeSize

	index = make([]int32, height*width)
	bounds = make([]int, 0, ((height+window-1)/window)*((width+window-1)/window)+1)
	bounds = append(bounds, 0)

	var cur int32
	for y := 0; y < height; y += window {
		for x := 0; x < width; x += window {
			h1 := min(window, height-y)
			w1 := min(window, width-x)
			for dy := range h1 {
				for dx := range w1 {
					win := (y+dy)*width + (x + dx)
					index[win] = cur
					cur++
				}
			}
			bounds = append(bounds, int(cur)*m.spatialMergeSize*m.spatialMergeSize)
		}
	}

	return index, bounds
}

func newVisionModel(c fs.Config) *VisionModel {
	blockCount := c.Uint("vision.block_count", 0)
	hiddenSize := int(c.Uint("vision.embedding_length", 0))
	numHeads := int(c.Uint("vision.attention.head_count", 1))
	numKVHeads := int(c.Uint("vision.attention.head_count_kv", uint32(numHeads)))
	patchSize := int(c.Uint("vision.patch_size", 14))
	spatialMergeSize := int(c.Uint("vision.spatial_merge_size", 2))

	return &VisionModel{
		Layers: make([]VisionEncoderLayer, blockCount),
		VisionModelOptions: &VisionModelOptions{
			hiddenSize:        hiddenSize,
			numHeads:          numHeads,
			numKVHeads:        numKVHeads,
			headDim:           hiddenSize / numHeads,
			patchSize:         patchSize,
			numChannels:       int(c.Uint("vision.num_channels", 3)),
			eps:               c.Float("vision.attention.layer_norm_epsilon", 1e-6),
			ropeTheta:         c.Float("vision.rope.freq_base", 10000),
			spatialMergeSize:  spatialMergeSize,
			windowSize:        int(c.Uint("vision.window_size", 112)),
			temporalPatchSize: int(c.Uint("vision.temporal_patch_size", 2)),
			fullAttnBlocks:    c.Ints("vision.fullatt_block_indexes", nil),
		},
	}
}
