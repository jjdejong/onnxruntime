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

#### Quick Build (No External Dependencies)

To build the Python wheel package for the system Python:

```bash
./build.sh --config Release \
  --build_wheel \
  --skip_tests \
  --parallel \
  --use_coreml \
  --use_xnnpack \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF
```

Install the wheel:
```bash
pip install build/MacOS/Release/dist/*.whl
```

#### Building for ComfyUI or Specific Python Environments

If you're building for a specific application like ComfyUI, Automatic1111, or any Python environment, follow this complete procedure to avoid dependency conflicts:

##### 1. Remove Conflicting Homebrew Packages

ONNX Runtime uses vendored dependencies (protobuf 21.12, ONNX). Remove system versions to avoid conflicts:

```bash
# Remove Homebrew ONNX (keeps other packages intact)
brew uninstall onnx --ignore-dependencies

# Optional: Check for protobuf conflicts
brew list | grep protobuf
# If found, consider uninstalling or the build will use vendored version
```

##### 2. Set Up Your Target Python Environment

Activate the Python environment where you'll install ONNX Runtime:

```bash
# For ComfyUI with venv:
cd /path/to/ComfyUI
source venv/bin/activate

# For conda:
conda activate your_env

# For system Python with specific version:
# Just note the Python path, e.g., /opt/homebrew/opt/python@3.12/bin/python3.12
```

##### 3. Install Build Dependencies in Target Environment

Install required packages in your activated environment:

```bash
pip install numpy wheel setuptools
```

**Note:** If using Homebrew Python and you get `externally-managed-environment` errors, you must use a virtual environment (venv/conda) or the application's existing environment.

##### 4. Clean Previous Build (If Exists)

```bash
cd /path/to/onnxruntime
rm -rf build
```

##### 5. Build the Wheel

With your target environment still activated:

```bash
./build.sh --config Release \
  --build_wheel \
  --skip_tests \
  --use_coreml \
  --use_xnnpack \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
  --cmake_extra_defines CMAKE_FIND_FRAMEWORK=LAST \
  --cmake_extra_defines CMAKE_FIND_APPBUNDLE=LAST \
  --parallel
```

**Key flags explained:**
- `--build_wheel`: Build Python package instead of just C++ library
- `--skip_tests`: Skip test phase (tests not needed for deployment)
- `--use_coreml`: Enable Neural Engine (NPU) support
- `--use_xnnpack`: Enable optimized CPU operations with KleidiAI
- `CMAKE_FIND_FRAMEWORK=LAST`: Prefer vendored dependencies over system
- `CMAKE_FIND_APPBUNDLE=LAST`: Prefer vendored dependencies over system

Build time: ~15-30 minutes depending on your Mac.

##### 6. Install the Wheel

```bash
# Still in your activated environment
pip install --force-reinstall build/MacOS/Release/dist/onnxruntime-*.whl
```

You may see dependency warnings like:
```
mediapipe 0.10.21 requires numpy<2, but you have numpy 2.2.6
```

These are usually harmless. Test the actual imports (see Verification below).

##### 7. Verification

**Important:** Change directory before testing (don't test from the onnxruntime source directory):

```bash
cd ~  # or cd /path/to/ComfyUI

python -c "
import onnxruntime as ort
print('✓ ONNX Runtime version:', ort.__version__)
print('✓ Available providers:', ort.get_available_providers())
"
```

**Expected output:**
```
✓ ONNX Runtime version: 1.24.0
✓ Available providers: ['CoreMLExecutionProvider', 'XnnpackExecutionProvider', 'CPUExecutionProvider']
```

If you see `CoreMLExecutionProvider`, you have NPU support! 🎉

##### 8. Test with Your Application

```bash
# For ComfyUI
cd /path/to/ComfyUI
python main.py

# For other applications, launch normally
```

##### 9. Monitor Neural Engine Usage (Optional)

While running inference, open another terminal:

```bash
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000
```

Look for `ane_power` activity to confirm Neural Engine usage.

##### Troubleshooting Python Wheel Builds

**Problem: `ModuleNotFoundError: No module named 'onnxruntime.capi'`**

Solution: You're in the source directory. Change to a different directory before testing:
```bash
cd ~
python -c "import onnxruntime; print(onnxruntime.__version__)"
```

**Problem: Wheel is wrong Python version (e.g., cp314 instead of cp312)**

Solution: The build used the wrong Python. Verify which Python is active:
```bash
which python
python --version
```

Activate your target environment *before* building.

**Problem: `Target "onnxruntime_pybind11_state" links to: Python::NumPy but the target was not found`**

Solution: NumPy not installed in the build environment:
```bash
pip install numpy wheel setuptools
```

**Problem: Protobuf version mismatch errors during build**

Solution: Remove Homebrew ONNX and use vendored dependencies:
```bash
brew uninstall onnx --ignore-dependencies
```

**Problem: Dependency conflicts after installation (numpy, protobuf)**

Solution: These warnings are usually safe to ignore. Test the actual imports:
```bash
python -c "import onnxruntime; import cv2; import mediapipe; print('All OK')"
```

If imports succeed, the warnings don't matter. If you need to resolve them:
- Upgrade conflicting packages: `pip install --upgrade mediapipe opencv-python`
- Or install a compatible numpy range: `pip install "numpy>=2.0,<2.3"`

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
