package onnx_test

import (
	"image"
	"testing"

	"github.com/deadelus/go-clean-onnxruntime/src/onnx"
)

func TestBoundingBox_RectArea(t *testing.T) {
	b := onnx.BoundingBox{X1: 0, Y1: 0, X2: 10, Y2: 10}
	if b.RectArea() != 100 {
		t.Errorf("Expected area 100, got %d", b.RectArea())
	}
}

func TestBoundingBox_ToRect(t *testing.T) {
	b := onnx.BoundingBox{X1: 1, Y1: 2, X2: 5, Y2: 6}
	rect := b.ToRect()
	expected := image.Rect(1, 2, 5, 6).Canon()
	if rect != expected {
		t.Errorf("Expected rect %v, got %v", expected, rect)
	}
}

func TestBoundingBox_Intersection_Union_IoU(t *testing.T) {
	b1 := onnx.BoundingBox{X1: 0, Y1: 0, X2: 10, Y2: 10}
	b2 := onnx.BoundingBox{X1: 5, Y1: 5, X2: 15, Y2: 15}
	inter := b1.Intersection(&b2)
	if inter <= 0 {
		t.Errorf("Expected intersection > 0, got %f", inter)
	}
	union := b1.Union(&b2)
	if union <= 0 {
		t.Errorf("Expected union > 0, got %f", union)
	}
	iou := b1.IoU(&b2)
	if iou <= 0 || iou >= 1 {
		t.Errorf("Expected IoU between 0 and 1, got %f", iou)
	}
}

func TestBoundingBox_ToString(t *testing.T) {
	b := onnx.BoundingBox{Label: "cat", Confidence: 0.9, X1: 1, Y1: 2, X2: 3, Y2: 4}
	s := b.ToString()
	if s == "" {
		t.Errorf("Expected non-empty string from ToString")
	}
}
