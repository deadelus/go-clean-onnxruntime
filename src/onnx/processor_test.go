package onnx_test

import (
	"image"
	"image/color"
	"testing"

	"github.com/deadelus/go-clean-onnxruntime/src/onnx"
)

func TestProcessorOutputFromData_SingleDetection(t *testing.T) {
	// Setup: 1 detection, 2 classes, threshold 0.5
	modelDetections := uint(1)
	modelOutputClasses := uint(2)
	modelClasses := []string{"cat", "dog"}
	width, height := 100, 100

	// Output tensor layout: [xc, yc, w, h, class1_prob, class2_prob]
	output := []float32{
		50, 50, 20, 20, // xc, yc, w, h
		0.1, 0.9, // class1_prob, class2_prob
	}

	p := &onnx.Processor{
		Image:               image.NewRGBA(image.Rect(0, 0, width, height)),
		ModelClasses:        modelClasses,
		ModelHeight:         uint(height),
		ModelWidth:          uint(width),
		ModelInputChannels:  3,
		ModelOutputClasses:  modelOutputClasses,
		ModelDetections:     modelDetections,
		ThresholdConfidence: 0.5,
	}

	boxes := p.OutputFromData(output)

	if len(boxes) != 1 {
		t.Fatalf("Expected 1 bounding box, got %d", len(boxes))
	}
	box := boxes[0]
	if box.Label != "dog" {
		t.Errorf("Expected label 'dog', got '%s'", box.Label)
	}
	if box.Confidence < 0.9 {
		t.Errorf("Expected confidence >= 0.9, got %f", box.Confidence)
	}
}

func TestProcessorInputToData_FillsTensor(t *testing.T) {
	width, height := 10, 10
	channels := 3
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	// Fill image with non-zero color using color.RGBA
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, &color.RGBA{R: 128, G: 64, B: 32, A: 255})
		}
	}
	modelDetections := uint(1)
	modelOutputClasses := uint(2)
	modelClasses := []string{"cat", "dog"}

	p := &onnx.Processor{
		Image:               img,
		ModelClasses:        modelClasses,
		ModelHeight:         uint(height),
		ModelWidth:          uint(width),
		ModelInputChannels:  uint(channels),
		ModelOutputClasses:  modelOutputClasses,
		ModelDetections:     modelDetections,
		ThresholdConfidence: 0.5,
	}

	channelSize := width * height
	data := make([]float32, channelSize*channels)

	err := p.InputToData(data)
	if err != nil {
		t.Fatalf("Processor.InputToData returned error: %v", err)
	}

	// Check that the tensor is filled (not all zeros)
	nonZero := false
	for _, v := range data {
		if v != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Errorf("Expected tensor data to be filled, but all values are zero")
	}
}
