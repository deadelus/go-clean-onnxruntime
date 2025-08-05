// Package example implements the YOLOv11s neural network for object detection.
package example

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"runtime"

	"github.com/deadelus/go-clean-onnxruntime/src/onnx"
)

var (
	// relPathToModel is the relative path to the YOLOv11s ONNX model file.
	relPathToModel = "src/example/yolo11s.onnx"
	// modelHeight and modelWidth are the dimensions to which input images are resized.
	// The YOLOv11s model expects input images to be 640x640 pixels.
	// https://docs.ultralytics.com/fr/tasks/detect/
	modelHeight = 640
	modelWidth  = 640
	// The model processes one image at a time
	batchSize = 1
	// modelInputChannels is the number of channels in the input image (3 for RGB).
	modelInputChannels = 3
	// modelClasses is the number of classes the model can detect.
	modelClasses = 80
	// modelDetections is the number of detections the model can output.
	modelDetections = 8400
	// thresholdConfidence is the minimum confidence threshold for detections.
	thresholdConfidence = 0.5
)

// Array of YOLOv8 class labels
// This is the list of classes that the YOLOv8 model can detect.
var yoloClasses = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
	"bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
	"tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket",
	"bottle", "wine glass", "cup",
	"fork", "knife", "spoon",
	"bowl",
	"banana", "apple",
	"sandwich",
	"orange", "broccoli", "carrot",
	"hot dog", "pizza", "donut", "cake",
	"chair", "couch", "potted plant",
	"bed", "dining table", "toilet",
	"tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone",
	"microwave", "oven", "toaster",
	"sink", "refrigerator",
	"book", "clock", "vase",
	"scissors", "teddy bear", "hair drier",
	"toothbrush",
}

// Yolo11sExample represents the YOLOv11s neural implementation.
type Yolo11sExample struct {
	Session *onnx.ONNXSession
}

// getOnnxLibrary returns the path to the shared library based on the current OS and architecture.
func getOnnxLibrary() string {
	var relPath string

	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			relPath = "libraries/win/onnxruntime.dll"
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			relPath = "libraries/osx/onnxruntime_arm64.dylib"
		}
		if runtime.GOARCH == "amd64" {
			relPath = "libraries/osx/onnxruntime_amd64.dylib"
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			relPath = "libraries/linux/onnxruntime_arm64.so"
		} else {
			relPath = "libraries/linux/onnxruntime.so"
		}
	}
	if relPath == "" {
		panic("Unable to find a version of the onnxruntime library supporting this system.")
	}

	// Resolve relPath relative to the directory of this file (vendor-safe)
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		panic("Unable to determine caller path for vendor compatibility")
	}
	baseDir := filepath.Dir(filename)
	return filepath.Join(baseDir, relPath)
}

// NewNeuralNetwork initializes the ONNX runtime for YOLOv11s model.
func NewNeuralNetwork() (*Yolo11sExample, error) {
	// Use current working directory as base
	cwd, err := os.Getwd()
	if err != nil {
		panic("Unable to determine working directory: " + err.Error())
	}
	modelPath := filepath.Join(cwd, relPathToModel)

	if err != nil {
		return nil, fmt.Errorf("failed to get model file path: %w", err)
	}

	onnxRuntime := onnx.NewOnnxRuntime(
		modelPath,
		getOnnxLibrary(),
		onnx.TensorInputShape{
			BatchSize: int64(batchSize),
			Channels:  int64(modelInputChannels),
			Height:    int64(modelHeight),
			Width:     int64(modelWidth),
		},
		onnx.TensorOutputShape{
			BatchSize:  int64(batchSize),
			Classes:    int64(modelClasses + 4), // 4 for bounding box coordinates
			Detections: int64(modelDetections),
		},
	)
	if onnxRuntime == nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime for YOLOv11s model at %s", modelPath)
	}

	// Create a new ONNX session
	session, err := onnx.NewONNXSession(onnxRuntime)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session for YOLOv11s model at %s: %w", modelPath, err)
	}
	return &Yolo11sExample{
		Session: session,
	}, nil
}

// AnalyzeImage implements the AI interface for Yolo11sExample.
func (m *Yolo11sExample) AnalyzeImage(image image.Image) ([]onnx.BoundingBox, error) {
	processor := &onnx.Processor{
		Image:               image,
		ModelClasses:        yoloClasses,
		ModelHeight:         uint(modelHeight),
		ModelWidth:          uint(modelWidth),
		ModelInputChannels:  uint(modelInputChannels),
		ModelOutputClasses:  uint(modelClasses),
		ModelDetections:     uint(modelDetections),
		ThresholdConfidence: float32(thresholdConfidence),
	}

	err := processor.Input(m.Session.TensorInput)

	if err != nil {
		return nil, fmt.Errorf("failed to process input: %w", err)
	}

	err = m.Session.Session.Run()
	if err != nil {
		return nil, fmt.Errorf("failed to run session: %w", err)
	}

	boxes := processor.Output(m.Session.TensorOutput)

	if boxes == nil {
		return nil, fmt.Errorf("no bounding boxes detected")
	}

	return boxes, nil
}
