package exaone4_5

import (
	"fmt"
	"image"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/model/imageproc"
)

type ImageProcessor struct {
	numChannels       int
	patchSize         int
	temporalPatchSize int
	mergeSize         int
	minPixels         int
	maxPixels         int
	factor            int
	imageMean         [3]float32
	imageStd          [3]float32
}

func newImageProcessor(c fs.Config) ImageProcessor {
	patchSize := int(c.Uint("vision.patch_size", 14))
	mergeSize := int(c.Uint("vision.spatial_merge_size", 2))

	return ImageProcessor{
		numChannels:       int(c.Uint("vision.num_channels", 3)),
		patchSize:         patchSize,
		temporalPatchSize: int(c.Uint("vision.temporal_patch_size", 2)),
		mergeSize:         mergeSize,
		minPixels:         int(c.Uint("vision.min_pixels", 56*56)),
		maxPixels:         int(c.Uint("vision.max_pixels", c.Uint("vision.longest_edge", 3211264))),
		factor:            patchSize * mergeSize,
		imageMean:         imageproc.ClipDefaultMean,
		imageStd:          imageproc.ClipDefaultSTD,
	}
}

func (p *ImageProcessor) SmartResize(height, width int) (int, int) {
	factor := p.factor

	if height < factor || width < factor {
		panic(fmt.Sprintf("height:%d or width:%d must be larger than factor:%d", height, width, factor))
	} else if aspectRatio := max(height, width) / min(height, width); aspectRatio > 200 {
		panic(fmt.Sprintf("absolute aspect ratio must be smaller than 200, got %v", aspectRatio))
	}

	round := func(x float64) int { return int(math.RoundToEven(x)) }

	hBar := round(float64(height)/float64(factor)) * factor
	wBar := round(float64(width)/float64(factor)) * factor

	if hBar*wBar > p.maxPixels {
		beta := math.Sqrt(float64(height*width) / float64(p.maxPixels))

		hBar = int(math.Floor(float64(height)/beta/float64(factor))) * factor
		wBar = int(math.Floor(float64(width)/beta/float64(factor))) * factor
	} else if hBar*wBar < p.minPixels {
		beta := math.Sqrt(float64(p.minPixels) / float64(height*width))

		hBar = int(math.Ceil(float64(height)*beta/float64(factor))) * factor
		wBar = int(math.Ceil(float64(width)*beta/float64(factor))) * factor
	}

	return hBar, wBar
}

type Grid struct {
	Height   int
	Width    int
	Temporal int
}

func (p *ImageProcessor) ProcessImage(img image.Image) ([]float32, *Grid, error) {
	img = imageproc.Composite(img)

	origWidth := img.Bounds().Dx()
	origHeight := img.Bounds().Dy()

	resizedHeight, resizedWidth := p.SmartResize(origHeight, origWidth)
	resizedImg := imageproc.Resize(img, image.Point{X: resizedWidth, Y: resizedHeight}, imageproc.ResizeBilinear)
	normalizedPixels := imageproc.Normalize(resizedImg, p.imageMean, p.imageStd, true, true)

	grid := &Grid{
		Height:   resizedHeight / p.patchSize,
		Width:    resizedWidth / p.patchSize,
		Temporal: 1,
	}

	patches, err := p.createPatches(normalizedPixels, resizedHeight, resizedWidth, grid)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create patches: %w", err)
	}

	return patches, grid, nil
}

func (p *ImageProcessor) createPatches(pixels []float32, height, width int, grid *Grid) ([]float32, error) {
	channels := p.numChannels
	patchSize := p.patchSize
	mergeSize := p.mergeSize
	temporalPatchSize := p.temporalPatchSize

	numPatches := grid.Temporal * grid.Height * grid.Width
	patchDim := channels * temporalPatchSize * patchSize * patchSize

	result := make([]float32, numPatches*patchDim)
	patchIndex := 0

	for range grid.Temporal {
		for h := 0; h < grid.Height; h += mergeSize {
			for w := 0; w < grid.Width; w += mergeSize {
				for mh := range mergeSize {
					for mw := range mergeSize {
						baseOffset := patchIndex * patchDim

						for c := range channels {
							channelOffset := baseOffset + (c * temporalPatchSize * patchSize * patchSize)

							for py := range patchSize {
								for px := range patchSize {
									y := (h+mh)*patchSize + py
									x := (w+mw)*patchSize + px
									srcIdx := c*height*width + y*width + x
									dstIdx := channelOffset + (py * patchSize) + px

									if srcIdx < len(pixels) && dstIdx < len(result) {
										result[dstIdx] = pixels[srcIdx]
									}
								}
							}
						}

						if temporalPatchSize > 1 {
							for c := range channels {
								channelOffset := baseOffset + (c * temporalPatchSize * patchSize * patchSize)
								frameSize := patchSize * patchSize
								firstFrameOffset := channelOffset

								for tp := 1; tp < temporalPatchSize; tp++ {
									currentFrameOffset := channelOffset + (tp * frameSize)
									copy(result[currentFrameOffset:currentFrameOffset+frameSize], result[firstFrameOffset:firstFrameOffset+frameSize])
								}
							}
						}

						patchIndex++
					}
				}
			}
		}
	}

	return result, nil
}
