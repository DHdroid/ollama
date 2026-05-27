package create

import (
	"os"
	"path/filepath"
	"strings"

	modeltypes "github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/safetensors"
)

var inlineMTPDraftArchitectures = map[string]string{
	"Exaone4_5_ForConditionalGeneration": "Exaone4_5_ForConditionalGenerationMTP",
}

func InlineMTPDraftMetadata(modelDir string) (*modeltypes.Draft, error) {
	cfg, err := readSourceModelConfig(modelDir)
	if err != nil {
		return nil, nil
	}
	arch := cfg.Architecture()
	draftArch, ok := inlineMTPDraftArchitectures[arch]
	if !ok {
		return nil, nil
	}

	transform, err := newTensorImportTransform(modelDir, cfg)
	if err != nil {
		return nil, err
	}

	entries, err := os.ReadDir(modelDir)
	if err != nil {
		return nil, err
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".safetensors") {
			continue
		}
		extractor, err := safetensors.OpenForExtraction(filepath.Join(modelDir, entry.Name()))
		if err != nil {
			return nil, err
		}
		for _, name := range extractor.ListTensors() {
			if strings.HasPrefix(name, "mtp.") && !transform.skipTensor(name) {
				extractor.Close()
				return &modeltypes.Draft{
					ModelFormat:  "safetensors",
					Architecture: draftArch,
					TensorPrefix: "mtp.",
					Config:       "config.json",
				}, nil
			}
		}
		extractor.Close()
	}
	return nil, nil
}
