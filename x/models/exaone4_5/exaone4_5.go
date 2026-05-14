// Package exaone4_5 registers EXAONE 4.5 architectures for MLX.
package exaone4_5

import (
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/exaone4"
)

func init() {
	base.Register("Exaone4_5_ForConditionalGeneration", exaone4.NewModel)
}
