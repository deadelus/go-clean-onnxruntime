package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"os"

	"github.com/deadelus/go-clean-onnxruntime/src/example"
)

const testImagePath = "src/example/assets/test_image.jpg"

func main() {
	fmt.Println("Hello, ONNX!")

	img, err := loadImageFromPath(testImagePath)
	if err != nil {
		fmt.Printf("Error loading image: %v\n", err)
		return
	}

	fmt.Printf("Loaded image: %v\n", img.Bounds())

	onnxExample, err := example.NewNeuralNetwork()
	if err != nil {
		fmt.Printf("Error creating neural network: %v\n", err)
		return
	}

	result, err := onnxExample.AnalyzeImage(img)
	if err != nil {
		fmt.Printf("Error analyzing image: %v\n", err)
		return
	}

	fmt.Printf("Analysis result: %v\n", result)
}

func loadImageFromPath(path string) (image.Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open image: %w", err)
	}
	defer file.Close()

	img, err := jpeg.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("failed to decode jpeg: %w", err)
	}
	return img, nil
}
