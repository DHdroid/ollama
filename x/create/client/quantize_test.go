package client

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestDecodeSourceFP8TensorAcceptsWeightScale(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX unavailable: %v", err)
	}

	weight := mlx.FromValues([]uint8{0, 1, 2, 3}, 2, 2)
	scale := mlx.FromValues([]float32{1}, 1, 1).AsType(mlx.DTypeBFloat16)
	if len(weight.Dims()) != 2 || len(scale.Dims()) != 2 {
		t.Skipf("MLX runtime unavailable: expected 2D fp8 weight and scale tensors, got %v and %v", weight.Dims(), scale.Dims())
	}
	got, err := decodeSourceFP8Tensor(weight, scale)
	if err != nil {
		t.Fatal(err)
	}
	mlx.Eval(got)
	if dims := got.Dims(); len(dims) != 2 || dims[0] != 2 || dims[1] != 2 {
		t.Fatalf("decoded dims = %v, want [2 2]", dims)
	}
}
