package create

import (
	"github.com/ollama/ollama/x/safetensors"
)

type exaone4ImportTransform struct {
	noopImportTransform
}

func newExaone4ImportTransform(string, sourceModelConfig) (tensorImportTransform, error) {
	return exaone4ImportTransform{}, nil
}

func (t exaone4ImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	return t.noopImportTransform.transformTensor(td)
}
