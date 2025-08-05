package onnx_test

import (
	"testing"

	"github.com/deadelus/go-clean-onnxruntime/src/onnx"
)

func TestONNXSession_MethodsCoverage(t *testing.T) {
	s := &onnx.ONNXSession{}
	s.Close()
	s.Close() // Should be idempotent

	// These methods require ONNX environment and valid library, so just check no panic occurs
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("SetInputTensor or SetOutputTensor panicked: %v", r)
		}
	}()
	s.SetInputTensor(onnx.TensorInputShape{BatchSize: 1, Channels: 3, Height: 2, Width: 2})
	s.SetOutputTensor(onnx.TensorOutputShape{BatchSize: 1, Classes: 2, Detections: 2})
}

func TestNewONNXSession_ErrorPath(t *testing.T) {
	nr := onnx.NewOnnxRuntime(
		"dummy.onnx",
		"missing_library.so",
		onnx.TensorInputShape{BatchSize: 1, Channels: 3, Height: 2, Width: 2},
		onnx.TensorOutputShape{BatchSize: 1, Classes: 2, Detections: 2},
	)
	_, err := onnx.NewONNXSession(nr)
	if err == nil {
		t.Errorf("Expected error when ONNX library is missing, but got nil")
	}
}

func TestNewONNXSession_SuccessPath(t *testing.T) {
	nr := onnx.NewOnnxRuntime(
		"src/example/yolo11s.onnx",
		"src/example/libraries/linux/onnxruntime.so",
		onnx.TensorInputShape{BatchSize: 1, Channels: 3, Height: 640, Width: 640},
		onnx.TensorOutputShape{BatchSize: 1, Classes: 84, Detections: 8400},
	)
	sess, err := onnx.NewONNXSession(nr)
	if err != nil {
		t.Skipf("Skipping: ONNX model or library not available: %v", err)
	}
	if sess == nil {
		t.Errorf("Expected session, got nil")
	}
}

// TestNewONNXSession_InvalidShape tests session creation with invalid tensor shapes.
func TestNewONNXSession_InvalidShape(t *testing.T) {
	nr := onnx.NewOnnxRuntime(
		"src/example/yolo11s.onnx",
		"src/example/libraries/linux/onnxruntime.so",
		onnx.TensorInputShape{BatchSize: 0, Channels: 0, Height: 0, Width: 0},
		onnx.TensorOutputShape{BatchSize: 0, Classes: 0, Detections: 0},
	)
	_, err := onnx.NewONNXSession(nr)
	if err == nil {
		t.Errorf("Expected error with invalid tensor shapes, got nil")
	}
}
