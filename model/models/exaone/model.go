package exaone

import (
	"strings"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/models/exaone4"
	exaone45 "github.com/ollama/ollama/model/models/exaone4_5"
)

func New(c fs.Config) (model.Model, error) {
	if Is45(c) {
		return exaone45.New(c)
	}
	return exaone4.New(c)
}

func Is45(c fs.Config) bool {
	// EXAONE 4.0 and 4.5 GGUFs both use general.architecture=exaone4,
	// so distinguish 4.5 by the model name metadata.
	name := strings.ToLower(c.String("general.basename") + " " + c.String("general.name"))
	return strings.Contains(name, "exaone-4.5")
}

func init() {
	model.Register("exaone4", New)
	model.Register("exaone4_5", exaone45.New)
}
