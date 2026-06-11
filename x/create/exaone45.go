package create

import (
	"strings"

	"github.com/ollama/ollama/x/safetensors"
)

type exaone45ImportTransform struct {
	noopImportTransform
}

func newExaone45ImportTransform(string, sourceModelConfig) (tensorImportTransform, error) {
	return exaone45ImportTransform{}, nil
}

func (exaone45ImportTransform) skipTensor(name string) bool {
	return strings.HasPrefix(name, "model.visual.") || strings.HasPrefix(name, "visual.")
}

func (t exaone45ImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	return t.noopImportTransform.transformTensor(td)
}
