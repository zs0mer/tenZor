# TenZor

A header-only C++ tensor library with CPU and CUDA GPU support.

## Features

- **Header-only** — just include `TenZor.hpp`
- **Custom slab allocator** (`Salloc`) — three-tier (small/medium/large), thread-local, cache-friendly
- **CPU & GPU tensors** — seamlessly move data between devices with `copyTo()`
- **OpenMP parallelism** — automatic multi-threading for large operations on CPU
- **cuBLAS acceleration** — `dot` and `matmul` use cuBLAS for `float` and `double`
- **Broadcasting** — NumPy-style broadcasting via zero strides
- **Configurable** — key constants and behaviors can be overridden at compile time
- **Extensible GPU functions** — add custom CUDA functors via `ExtraGPUFunctions.hpp`

## Requirements

- GCC 11+
- CUDA Toolkit
- OpenMP
- [doctest](https://github.com/doctest/doctest) (for tests only)
- CMake 3.16+

With Nix, all dependencies are provided by the flake:

```sh
nix develop
```

## Building

```sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DTESTS=OFF
make -j$(nproc)
```

For a debug build:

```sh
cmake .. -DCMAKE_BUILD_TYPE=Debug -DTESTS=OFF
```
if you don't have CUDA, then also do ```-DUSE_CUDA=OFF```

## Running Tests

From the build directory:

```sh
make tests
```

Or run individual test binaries:

```sh
./Tallocator
./Tbuffer
./Ttensor
./Tother_classes
./Toperations
./Tgpu_function
```

## Usage

Include the main header:

```cpp
#include "Tenzor.hpp"
```

### Basic Types

| Type | Description |
|---|---|
| `tz::Scalar<T>` | 0-dimensional tensor (single value) |
| `tz::Vector<T>` | 1-dimensional tensor |
| `tz::Matrix<T>` | 2-dimensional tensor |
| `tz::Tensor<T>` | N-dimensional tensor (up to `MAX_DIM`, default 4) |

### Creating Tensors

```cpp
using namespace tz;

// From initializer lists
Vector<float> v = {1.f, 2.f, 3.f};
Matrix<float> m = {{1.f, 2.f}, {3.f, 4.f}};
Scalar<int>   s(42);

// By shape and device
Vector<float> v2(128, CPU);
Matrix<float> m2(4, 4, GPU);

// From std::vector (nested)
std::vector<std::vector<float>> data = {{1.f, 2.f}, {3.f, 4.f}};
Tensor<float> t = Tensor<float>::fromSTDVec(data);
```

### Indexing

```cpp
Matrix<int> m = {{1, 2, 3}, {4, 5, 6}};

m.at(0, 1);       // 2
m.row(0);         // Vector view of row 0
m.col(2);         // Vector view of col 2  — non-owning, modifying it modifies m
m[1];             // Vector view of row 1
m.transpose();    // Transposed view — no copy
```

### Arithmetic

```cpp
Vector<float> a = {1.f, 2.f, 3.f};
Vector<float> b = {4.f, 5.f, 6.f};
Scalar<float> s(2.f);

Vector<float> c = a + b;
Vector<float> d = a * s;
a += b;
a.setAll(0.f);

Scalar<float> r  = dot(a, b);
Matrix<float> mm = matmul(m1, m2);
Scalar<double> d = det(sq);     // Gaussian elimination, CPU only
```

### Moving Between Devices

```cpp
Vector<float> cpu_v = {1.f, 2.f, 3.f};
Vector<float> gpu_v = cpu_v.copyTo(GPU);   // copies to GPU

Vector<float> back  = gpu_v.copyTo(CPU);   // copies back
```

### Broadcasting

```cpp
Scalar<int> s(7);
Matrix<int> m = s.broadcast(4, 4);    // 4x4 matrix where every element is 7

Vector<int> v = {1, 2, 3};
Matrix<int> bcast = v.broadcast(5, 3); // each of 5 rows is {1, 2, 3}
```

### Apply (custom element-wise operations)

`apply` is the core primitive for element-wise operations. It works on both CPU and GPU.

```cpp
// Unary: func(a[i])
TensorIMPL<float>::apply(t, MyFunc{});

// Binary: func(a[i], b[i])  — b is writable
TensorIMPL<float>::apply(a, b, MyFunc{});

// Ternary: func(a[i], b[i], c[i])  — c is writable
TensorIMPL<float>::apply(a, b, c, MyFunc{});
```

To allow GPU execution, pass `tz::impl::AnyDevice{}` as the last argument (default is `CPUOnly`):

```cpp
TensorIMPL<float>::apply(gpu_tensor, MyFunc{}, tz::impl::AnyDevice{});
```

### Adding Custom GPU Functions

Define your functors in `Demo/src/ExtraFunctions.hpp` (or wherever you point `include/ExtraGPUFunctions.hpp`), then instantiate them with the provided macros:

```cpp
template <class T>
struct MyFunc {
    TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
        b = a * a;
    }
};

#define TZ_EXTRA_GPU_FUNCTIONS_INSTANTIATE(T) \
    INSTANTIATE_BINARY_APPLY(T, MyFunc);
```

Available instantiation macros:

```
INSTANTIATE_UNARY_APPLY(T, Func)
INSTANTIATE_BINARY_APPLY(T, Func)
INSTANTIATE_TERNARY_APPLY(T, Func)
```

## Configuration

The following can be set as compile-time defines (e.g. via `-DTZ_MAX_DIM=8`), or defined before including `TenZor.hpp`:

| Define | Default | Description |
|---|---|---|
| `TZ_MAX_DIM` | `4` | Maximum number of tensor dimensions |
| `TZ_START_MEM_SIZE` | `10 * 1024 * 1024` | Initial pool size for `Salloc` (bytes) |
| `TZ_DEFAULT_ALIGNMENT` | `64` | Default memory alignment |
| `TZ_CTD` | `1` | Allow cross-thread deallocation (sequential only — both threads must not run simultaneously) |
| `TZ_ERRORS` | `1` | Enable runtime error checking (throws `std::runtime_error`) |
| `TZ_IMMUTABLE_BROADCASTS` | `1` | Disallow `at()` and `apply()` on broadcasted tensors |
| `TZ_APPLY_ERROR_IF_SHAPE_NOT_SAME` | `1` | Error instead of reshape in `apply()` when shapes differ |
| `TZ_DEFAULT_ALLOCATOR` | See below | Override the default allocator selection logic |

Default allocator logic:
```cpp
if (device == CPU) return Salloc::instance();
if (device == GPU) return Galloc::instance();
return Malloc::instance();
```

## Allocators

| Allocator | Device | Description |
|---|---|---|
| `tz::mem::Salloc` | CPU | Thread-local slab allocator. Fast for all sizes. Singleton. |
| `tz::mem::Malloc` | CPU | Thin wrapper around `aligned_alloc` / `free`. Singleton. |
| `tz::mem::Galloc` | GPU | Wrapper around `cudaMalloc` / `cudaFree`. Singleton. |

All implement the `tz::mem::Allocator` interface and can be passed anywhere an allocator is expected.

## Known Limitations

- Classes cannot be used in global memory (static storage) due to thread-local allocator initialization order
- `det()` is CPU-only and is not reliable for integer types
- Cross-thread deallocation is supported sequentially (`TZ_CTD=1`) but **not** concurrently — both threads must not run simultaneously when one deallocates memory allocated by the other
- `Salloc` maximum alignment is 64 bytes
- `matmul` on GPU is only implemented for `float` and `double`
