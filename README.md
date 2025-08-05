# go-clean-onnxruntime

[![Go Version](https://img.shields.io/badge/go-1.21+-blue.svg)](https://golang.org/dl/)
[![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen.svg)](https://pkg.go.dev/testing)
[![Lint](https://badgen.net/badge/lint/ok/green?icon=https://raw.githubusercontent.com/essentialkaos/go-badge/master/.github/images/golangci-lint.svg)](https://github.com/essentialkaos/go-badge)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**go-clean-onnxruntime** is a clean, vendorizable fork of [github.com/yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go)  
**Upstream version:** Based on [github.com/yalue/onnxruntime_go v1.21.0](https://github.com/yalue/onnxruntime_go/releases/tag/v1.21.0)

This project refactors and simplifies the original library for easier integration, improved path handling, and full vendor compatibility.

---

## Features

- Clean Go API for ONNX model inference
- Fully vendorizable (no absolute paths, works in vendor/)
- Refactored and simplified from the original yalue/onnxruntime_go
- Process and analyze images (JPEG)
- Example neural network integration
- High test coverage

---

## Project Origin

This library is based on [github.com/yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go)  
at version [v1.21.0](https://github.com/yalue/onnxruntime_go/releases/tag/v1.21.0).  
It includes significant improvements for vendor compatibility, code clarity, and ease of use in modern Go projects.

---

## Installation

Clone the repository and use as a Go module:

```sh
git clone https://github.com/deadelus/go-clean-onnxruntime.git
cd go-clean-onnxruntime
go mod tidy
```

---

## Usage Example

Below is a minimal example that loads a JPEG image, initializes a neural network, and analyzes the image:

```go
package main

import (
    "deadelus/go-clean-onnxruntime/src/onnx/example"
    "fmt"
    "image"
    "image/jpeg"
    "os"
)

const testImagePath = "src/onnx/assets/test_image.jpg"

func main() {
    // Load an image from disk
    img, err := loadImageFromPath(testImagePath)
    if err != nil {
        fmt.Printf("Error loading image: %v\n", err)
        return
    }

    // Create a new ONNX neural network instance
    onnxExample, err := example.NewNeuralNetwork()
    if err != nil {
        fmt.Printf("Error creating neural network: %v\n", err)
        return
    }

    // Analyze the image using the neural network
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
```

---

## How It Works

1. **Image Loading:**  
   The example loads a JPEG image from disk using Go's standard library.

2. **Model Initialization:**  
   `example.NewNeuralNetwork()` creates an ONNX model instance ready for inference.

3. **Image Analysis:**  
   The loaded image is passed to `AnalyzeImage`, which runs inference and returns results.

---

## Testing & Coverage

Run all tests and view coverage:

```sh
go test ./src/onnx/... -cover
```

Coverage: **88%**

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.