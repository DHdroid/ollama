package create

import (
	"strings"

	"github.com/ollama/ollama/x/safetensors"
)

type exaone4ImportTransform struct {
	noopImportTransform
}

func newExaone4ImportTransform(string, sourceModelConfig) (tensorImportTransform, error) {
	return exaone4ImportTransform{}, nil
}

func (exaone4ImportTransform) skipTensor(name string) bool {
	return strings.HasPrefix(name, "mtp.") || strings.HasPrefix(name, "model.visual.") || strings.HasPrefix(name, "visual.")
}

func (t exaone4ImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	return t.noopImportTransform.transformTensor(td)
}
