# Building ONNX Runtime for macOS ARM64 (Apple Silicon)

This guide covers building ONNX Runtime on Apple Silicon Macs (M1, M2, M3, and later).

## Prerequisites

### Required Tools
- **Xcode Command Line Tools** (or full Xcode)
  ```bash
  xcode-select --install
  ```
- **CMake** 3.26 or higher
  ```bash
  brew install cmake
  ```
- **Python** 3.8 or higher
  ```bash
  brew install python@3.11
  ```

### Optional Tools
- **Ninja** build system (recommended for faster builds)
  ```bash
  brew install ninja
  ```

## Quick Start

### Basic Build
The simplest way to build ONNX Runtime for Apple Silicon:

```bash
./build.sh --config Release --build_shared_lib --parallel
```

This will:
- Build for ARM64 (native architecture)
- Create a Release build
- Generate a shared library (`libonnxruntime.dylib`)
- Use parallel compilation

### Recommended Build with Optimizations

For optimal performance on Apple Silicon, enable additional optimizations:

```bash
./build.sh --config Release \
  --build_shared_lib \
  --parallel \
  --use_coreml \
  --use_xnnpack \
  --enable_arm_neon_nchwc \
  --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64
```

**Optimizations enabled:**
- `--use_coreml`: Enable CoreML execution provider (uses Apple Neural Engine)
- `--use_xnnpack`: Enable XNNPACK execution provider (optimized CPU operations)
- `--enable_arm_neon_nchwc`: Enable ARM NEON NCHWc kernels (good for high thread counts)

## Apple Silicon Specific Optimizations

ONNX Runtime includes several Apple Silicon-specific optimizations:

### 1. MLAS Optimizations (Enabled by Default)
The Microsoft Linear Algebra Subprograms (MLAS) library includes ARM64-optimized kernels:
- **FP16 (Half-precision)**: Optimized half-precision GEMM operations
- **I8MM (Int8 Matrix Multiply)**: SMMLA/UMMLA instructions for quantized models
- **Dot Product**: UDOT/SDOT instructions for improved convolution performance
- **BF16 (Brain Float 16)**: Available on M2/M3 and later

These optimizations are automatically compiled and enabled at runtime based on CPU capabilities.

### 2. CoreML Execution Provider
The CoreML EP leverages Apple's Neural Engine for accelerated inference:

```bash
./build.sh --config Release --use_coreml --build_shared_lib
```

**Features:**
- Automatic offloading to Neural Engine when supported
- Supports both CoreML 4+ (MLProgram) and legacy (NeuralNetwork) formats
- Fallback to CPU for unsupported operators

See supported operators:
- [CoreML MLProgram operators](../tools/ci_build/github/apple/coreml_supported_mlprogram_ops.md)
- [CoreML NeuralNetwork operators](../tools/ci_build/github/apple/coreml_supported_neuralnetwork_ops.md)

### 3. XNNPACK Execution Provider
XNNPACK provides highly optimized operators for ARM CPUs:

```bash
./build.sh --config Release --use_xnnpack --build_shared_lib
```

**Features:**
- Optimized for NHWC (channels-last) layout
- Includes KleidiAI optimizations for ARM64
- No concurrent execution (sequential execution only)

## Build Configurations

### Universal Binary (ARM64 + x86_64)
To build a universal binary that runs on both Apple Silicon and Intel Macs:

```bash
./build.sh --config Release \
  --build_shared_lib \
  --parallel \
  --cmake_extra_defines CMAKE_OSX_ARCHITECTURES="arm64;x86_64"
```

**Note:** Cross-compilation from ARM64 to x86_64 requires Rosetta 2:
```bash
softwareupdate --install-rosetta
```

### Debug Build
For development and debugging:

```bash
./build.sh --config Debug \
  --build_shared_lib \
  --parallel \
  --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64
```

### Minimal Build
For size-constrained environments:

```bash
./build.sh --config MinSizeRel \
  --build_shared_lib \
  --parallel \
  --minimal_build \
  --disable_exceptions \
  --disable_rtti
```

## Advanced Build Options

### Python Wheel
To build the Python wheel package:

```bash
./build.sh --config Release \
  --build_wheel \
  --parallel \
  --use_coreml \
  --use_xnnpack
```

Install the wheel:
```bash
pip install build/MacOS/Release/dist/*.whl
```

### Enable All Execution Providers
```bash
./build.sh --config Release \
  --build_shared_lib \
  --parallel \
  --use_coreml \
  --use_xnnpack \
  --use_dnnl \
  --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64
```

### Static Library
To build a static library instead of shared:

```bash
./build.sh --config Release \
  --parallel \
  --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64
```

## Performance Tuning

### Thread Count
ONNX Runtime automatically detects CPU cores. To override:

```cpp
SessionOptions session_options;
session_options.SetIntraOpNumThreads(8);  // Set to number of performance cores
```

On Apple Silicon:
- **M1**: 4 performance cores + 4 efficiency cores
- **M2**: 4 performance cores + 4 efficiency cores
- **M3**: 4 performance cores + 4 efficiency cores
- **M1/M2/M3 Pro**: 6-12 performance cores + 4 efficiency cores
- **M1/M2/M3 Max/Ultra**: 8-16 performance cores + 4-8 efficiency cores

For best performance, set thread count to the number of **performance cores only**.

### Execution Provider Priority
When multiple execution providers are enabled, set priority:

```cpp
SessionOptions session_options;
std::vector<std::string> providers = {"CoreMLExecutionProvider", "CPUExecutionProvider"};
session = Ort::Session(env, model_path, session_options, providers);
```

## Testing

### Run Unit Tests
```bash
./build.sh --config Release --build_shared_lib --parallel --test
```

### Run Specific Test
```bash
./build/MacOS/Release/onnxruntime_test_all --gtest_filter=*TestName*
```

## Troubleshooting

### Build Fails with "Unknown Architecture"
Ensure you're using CMake 3.19.2 or higher, which properly supports Apple Silicon:
```bash
cmake --version
brew upgrade cmake
```

### Linker Errors with FP16/I8MM
These optimizations require ARMv8.2-a or later. All Apple Silicon chips support this, but ensure you're not cross-compiling incorrectly.

### CoreML EP Not Found at Runtime
Ensure the shared library was built with `--use_coreml` and that the model contains supported operators.

### Performance Lower Than Expected
1. Check that you're running natively (not under Rosetta):
   ```bash
   arch
   # Should output: arm64
   ```

2. Verify execution provider is being used:
   ```cpp
   session_options.SetLogSeverityLevel(0);  // Enable verbose logging
   ```

3. Profile with Instruments (Xcode):
   ```bash
   instruments -t "Time Profiler" ./your_app
   ```

## Additional Resources

- [Build script options](../tools/ci_build/build.py) - Full list of build flags
- [Execution Providers](https://onnxruntime.ai/docs/execution-providers/) - EP documentation
- [Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance.html) - General performance guide
- [GitHub Actions macOS Workflow](../.github/workflows/mac.yml) - CI/CD examples

## Known Limitations

1. **FP16 Models**: While FP16 kernels are enabled, end-to-end FP16 model support may vary by operator
2. **Metal EP**: Not yet implemented (CoreML provides GPU acceleration via Neural Engine)
3. **Accelerate.framework**: Future integration planned for additional BLAS optimizations
4. **AMX Instructions**: Apple's Matrix coprocessor not yet directly utilized

## Future Enhancements

Planned optimizations for Apple Silicon:
- Integration with Apple's Accelerate.framework for optimized BLAS operations
- Metal Performance Shaders (MPS) execution provider for GPU compute
- Direct AMX (Apple Matrix) instruction support
- Enhanced quantized model support with I8MM optimizations

---

**Last Updated**: November 2025
**Applies to**: ONNX Runtime 1.x and later on macOS 11+ (Apple Silicon)
