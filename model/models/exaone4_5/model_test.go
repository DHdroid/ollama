package exaone4_5

import (
	"testing"

	"github.com/ollama/ollama/ml/backend/ggml"
	"github.com/ollama/ollama/model/input"
)

type fakeTensor struct {
	*ggml.Tensor
	dims []int
}

func (t *fakeTensor) Dim(i int) int {
	return t.dims[i]
}

func TestPostTokenizeVisionSpan(t *testing.T) {
	m := &Model{
		imageToken:  67,
		visionStart: 73,
		visionEnd:   74,
	}

	inputs := []*input.Input{
		{Token: 11},
		{
			Multimodal: []input.Multimodal{{
				Tensor: &fakeTensor{dims: []int{5120, 4, 1, 1}},
				Data:   &Grid{Width: 4, Height: 4},
			}},
			MultimodalHash: 123,
		},
		{Token: 12},
	}

	got, err := m.PostTokenize(inputs)
	if err != nil {
		t.Fatalf("PostTokenize() error = %v", err)
	}

	want := []struct {
		token     int32
		hash      uint64
		sameBatch int
		hasMM     bool
	}{
		{token: 11},
		{token: 73},
		{token: 67, hash: 123, sameBatch: 4, hasMM: true},
		{token: 67},
		{token: 67},
		{token: 67},
		{token: 74},
		{token: 12},
	}

	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}

	for i := range want {
		if got[i].Token != want[i].token {
			t.Fatalf("got[%d].Token = %d, want %d", i, got[i].Token, want[i].token)
		}
		if got[i].MultimodalHash != want[i].hash {
			t.Fatalf("got[%d].MultimodalHash = %d, want %d", i, got[i].MultimodalHash, want[i].hash)
		}
		if got[i].SameBatch != want[i].sameBatch {
			t.Fatalf("got[%d].SameBatch = %d, want %d", i, got[i].SameBatch, want[i].sameBatch)
		}
		hasMM := len(got[i].Multimodal) > 0
		if hasMM != want[i].hasMM {
			t.Fatalf("got[%d].hasMM = %v, want %v", i, hasMM, want[i].hasMM)
		}
	}
}
