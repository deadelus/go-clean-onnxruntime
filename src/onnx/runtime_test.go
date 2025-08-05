package onnx_test

import (
	"testing"

	"github.com/deadelus/go-clean-onnxruntime/src/onnx"
)

func TestNewOnnxRuntimeAndGetters(t *testing.T) {
	modelPath := "model.onnx"
	libraryPath := "libonnx.so"
	inputShape := onnx.TensorInputShape{BatchSize: 1, Channels: 3, Height: 224, Width: 224}
	outputShape := onnx.TensorOutputShape{BatchSize: 1, Classes: 80, Detections: 100}

	r := onnx.NewOnnxRuntime(modelPath, libraryPath, inputShape, outputShape)

	if r.GetModelPath() != modelPath {
		t.Errorf("GetModelPath() = %s, want %s", r.GetModelPath(), modelPath)
	}
	if r.GetLibraryPath() != libraryPath {
		t.Errorf("GetLibraryPath() = %s, want %s", r.GetLibraryPath(), libraryPath)
	}
	if r.GetTensorInputShape() != inputShape {
		t.Errorf("GetTensorInputShape() = %+v, want %+v", r.GetTensorInputShape(), inputShape)
	}
	if r.GetTensorOutputShape() != outputShape {
		t.Errorf("GetTensorOutputShape() = %+v, want %+v", r.GetTensorOutputShape(), outputShape)
	}
}

// TestNewOnnxRuntime_EdgeCases covers edge cases for OnnxRuntime.
func TestNewOnnxRuntime_EdgeCases(t *testing.T) {
	r := onnx.NewOnnxRuntime("", "", onnx.TensorInputShape{}, onnx.TensorOutputShape{})
	if r.GetModelPath() != "" {
		t.Errorf("Expected empty model path, got %s", r.GetModelPath())
	}
	if r.GetLibraryPath() != "" {
		t.Errorf("Expected empty library path, got %s", r.GetLibraryPath())
	}
	emptyInput := onnx.TensorInputShape{}
	if r.GetTensorInputShape() != emptyInput {
		t.Errorf("Expected empty input shape, got %+v", r.GetTensorInputShape())
	}
	emptyOutput := onnx.TensorOutputShape{}
	if r.GetTensorOutputShape() != emptyOutput {
		t.Errorf("Expected empty output shape, got %+v", r.GetTensorOutputShape())
	}
}
