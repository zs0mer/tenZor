---

# TenZor

A lightweight C++17 tensor and automatic differentiation library with CPU and CUDA support.

TenZor provides multidimensional tensors, its operations, a custom CPU allocator, CUDA kernels, basic neural-network layers, and optimizers.

## Features

- **CPU and CUDA tensors** (with explicit device transfers)
- **Row-major matrices** with conventional `matrix[row][column]` indexing
- **Tensor views** for indexing, rows, columns, transposes, and broadcasting
- **Reverse-mode automatic differentiation**
- **Neural-network utilities**
    - Linear layers
    - SGD
    - SGD with momentum
    - RMSProp
    - Adam
- **Custom slab allocator**
- **OpenMP parallelism** for CPU operations
- **cuBLAS acceleration** for GPU `dot` and `matmul`
- **Custom CPU/GPU element-wise functions** through `apply`

Most tensor functionality is implemented in headers. CUDA support additionally requires compiling and linking the CUDA backend.

## Requirements

### CPU

- C++17 compiler
- GCC 11
- CMake 3.16
- OpenMP

### CUDA

- CUDA Toolkit
- cuBLAS
- A supported NVIDIA GPU

### Tests

- [doctest](https://github.com/doctest/doctest)

### Nix

With Nix, the development dependencies are provided by the included flake:

```sh
nix develop
```

## Building

### CUDA build

CUDA and tests are enabled by default:

```sh
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DTESTS=ON

cmake --build build -j"$(nproc)"
```

### CPU-only build

```sh
cmake -S . -B build-cpu \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=OFF \
    -DTESTS=OFF

cmake --build build-cpu -j"$(nproc)"
```

The current CMake configuration registers the test suite only when CUDA is enabled.

### Debug build

```sh
cmake -S . -B build-debug \
    -DCMAKE_BUILD_TYPE=Debug \
    -DUSE_CUDA=ON \
    -DTESTS=ON

cmake --build build-debug -j"$(nproc)"
```
if you don't have CUDA, then also do ```-DUSE_CUDA=OFF```

## Running Tests

Run from inside the build directory:

```sh
make test
```

The test suite currently contains:

```text
allocator
buffer
tensor
other_classes
operations
gpu_function
autograd
```

Individual test binaries can also be run directly:

```sh
./build/Tallocator
./build/Tbuffer
./build/Ttensor
./build/Tother_classes
./build/Toperations
./build/Tgpu_function
./build/Tautograd
```

## Usage

Include the main header:

```cpp
#include "TenZor.hpp"
```

The public API is under the `tz` namespace:

```cpp
using namespace tz;
```

## Basic Types

| Type                      | Description                                  |
| ------------------------- | -------------------------------------------- |
| `tz::Scalar<T>`           | Zero-dimensional tensor containing one value |
| `tz::Vector<T>`           | One-dimensional tensor                       |
| `tz::Matrix<T>`           | Two-dimensional row-major tensor             |
| `tz::Tensor<T>`           | N-dimensional tensor                         |
| `tz::grad::GradScalar<T>` | Differentiable scalar                        |
| `tz::grad::GradVector<T>` | Differentiable vector                        |
| `tz::grad::GradMatrix<T>` | Differentiable matrix                        |
| `tz::grad::GradTensor<T>` | Differentiable N-dimensional tensor          |

The maximum number of tensor dimensions is controlled by `TZ_MAX_DIM` and defaults to `4`.

## Creating Tensors

```cpp
using namespace tz;

Scalar<float> scalar(3.5f);
Vector<float> vector = {
    1.f, 2.f, 3.f
};
Matrix<float> matrix = {
    {1.f, 2.f, 3.f},
    {4.f, 5.f, 6.f}
};
```

Initializer lists describe matrix rows, so the matrix above has shape:

```text
[2 rows, 3 columns]
```

Tensors can also be created by shape and device:

```cpp
Vector<float> vector(128, CPU);
Matrix<float> matrix(64, 128, GPU); // 64 rows, 128 columns
```

Nested `std::vector` values can be converted to a general tensor:

```cpp
std::vector<std::vector<float>> data = {
    {1.f, 2.f},
    {3.f, 4.f}
};

Tensor<float> tensor = Tensor<float>::fromSTDVec(data);
```

Nested containers and matrix initializer lists are **required** to be rectangular.

## Matrix and Vector Conventions

TenZor uses conventional row-based matrix indexing:

```cpp
matrix[row][column]
matrix.at(row, column)
```

For example:

```cpp
Matrix<int> matrix = {
    {1, 2, 3},
    {4, 5, 6}
};

matrix.at(0, 1); // 2
matrix.at(1, 2); // 6

matrix[0];       // {1, 2, 3}
matrix.row(1);   // {4, 5, 6}
matrix.col(2);   // {3, 6}
```

Rows, columns, indexing results, and transposes are views. They share memory with the original matrix:

```cpp
auto row = matrix.row(0);
row.at(1) = 10;

std::cout << matrix.at(0, 1); // 10
```

### Vector orientation

A `Vector<T>` is represented as a rank-one tensor. When converted to a matrix, it is interpreted as a column vector:

```text
Vector(N) -> Matrix [N, 1]
```

Its transpose is a row matrix:

```text
vector.transpose() -> Matrix [1, N]
```

Therefore, matrix-vector multiplication uses the conventional column-vector form:

\[
y = Wx
\]

with:

```text
W: [out, in]
x: [in, 1]
y: [out, 1]
```

## Transpose

Matrix transpose returns a view and does not copy the underlying data:

```cpp
Matrix<int> matrix = {
    {1, 2, 3},
    {4, 5, 6}
};

Matrix<int> transposed = matrix.transpose();
```

Shapes:

```text
matrix:     [2, 3]
transposed: [3, 2]
```

Values:

```text
matrix:
1 2 3
4 5 6

transposed:
1 4
2 5
3 6
```

Because the result is a view, modifying it also modifies the original matrix.

## Arithmetic

Element-wise operators are available for tensors with matching shapes:

```cpp
Vector<float> a = {1.f, 2.f, 3.f};
Vector<float> b = {4.f, 5.f, 6.f};

Vector<float> sum        = a + b;
Vector<float> difference = a - b;
Vector<float> product    = a * b;
Vector<float> quotient   = a / b;

a += b;
a -= b;
a *= b;
```

Scalar operations are also supported:

```cpp
Scalar<float> scale(2.f);

Vector<float> doubled = a * scale;
Vector<float> shifted = a + scale;

a *= scale;
```

Set all elements to a value:

```cpp
a.setAll(0.f);
```

Other element-wise functions include:

```cpp
auto exponentials = a.exp();
auto logarithms   = a.log();
auto powers       = a.pow(Scalar<float>(2.f));
```

## Reductions and Linear Algebra

### Sum

```cpp
Vector<float> values = {1.f, 2.f, 3.f};
Scalar<float> total = values.sum();
```

### Dot product

```cpp
Vector<float> a = {1.f, 2.f, 3.f};
Vector<float> b = {4.f, 5.f, 6.f};

Scalar<float> result = dot(a, b);
```

### Matrix multiplication

For:

```text
A: [M, K]
B: [K, N]
```

`matmul(A, B)` returns:

```text
C: [M, N]
```

Example:

```cpp
Matrix<float> a = {
    {1.f, 2.f, 3.f},
    {4.f, 5.f, 6.f}
};

Matrix<float> b = {
    {7.f,  8.f},
    {9.f, 10.f},
    {11.f, 12.f}
};

Matrix<float> c = matmul(a, b);
```

Result:

```text
58  64
139 154
```

GPU `matmul` is accelerated through cuBLAS for `float` and `double`.

## Moving Between Devices

Use `copyTo()` to explicitly transfer tensor data:

```cpp
Vector<float> cpu = {1.f, 2.f, 3.f};

Vector<float> gpu = cpu.copyTo(GPU);
Vector<float> back = gpu.copyTo(CPU);
```

`copyTo()` creates a new tensor. The source tensor remains on its original device.

Host element access through `at()` or `get()` requires CPU memory:

```cpp
float value = gpu.copyTo(CPU).at(0);
```

## Broadcasting

Broadcasting creates views using zero strides rather than copying values.

Broadcast a scalar:

```cpp
Scalar<int> scalar(7);

Matrix<int> matrix = scalar.broadcast(4, 4);
```

Broadcast a vector over matrix rows:

```cpp
Vector<int> vector = {1, 2, 3};

Matrix<int> matrix = vector.broadcast(5, 3);
```

The result is:

```text
1 2 3
1 2 3
1 2 3
1 2 3
1 2 3
```

Broadcasted tensors may alias the same underlying element multiple times. By default, writes through broadcasted views are disabled.

## Automatic Differentiation

Automatic differentiation is available under `tz::grad`:

```cpp
using namespace tz;
using namespace tz::grad;

GradScalar<float> a(Scalar<float>(3.f));
GradScalar<float> b(Scalar<float>(4.f));

auto result = fn::multiply<float>(a, b);
result.backward();
```

After backward:

```text
a.grad() = 4
b.grad() = 3
```

It is implemented for a wide range of operations

Autograd graphs are consumed by `backward()`. All operands and intermediate differentiable values referenced by a graph must remain alive until backward has completed.

## Neural-Network Layers

### Linear layer

`Linear<T>(in, out)` stores:

```text
weights: [out, in]
bias:    [out]
```

For a single input vector, it computes:

\[
y = Wx + b
\]

Example:

```cpp
using namespace tz;
using namespace tz::grad;

Linear<float> layer(3, 2);

GradVector<float> input(
    Vector<float>{1.f, 2.f, 3.f}
);

GradVector<float> output = layer(input);
```

## Optimizers

Available optimizers are:

- `SGD`
- `SGDmomentum`
- `RMSProp`
- `Adam`

Example:

```cpp
using namespace tz;
using namespace tz::grad;

Linear<float> layer(3, 2);
SGD<float> optimizer(layer.parameters(), 0.01f);

GradVector<float> input(
    Vector<float>{1.f, 2.f, 3.f}
);

auto output = layer(input);
auto loss = fn::sum<float>(output);

loss.backward();
optimizer.step();
```

`step()` updates the parameters and clears their accumulated gradients.

## Apply

`apply` is TenZor’s core element-wise execution primitive. It supports dense and strided tensors on both CPU and GPU. With a CUDA compiler, custom GPU operations can be written as small functors—much like compute shaders— while TenZor handles kernel launches, indexing, and tensor strides.

### Unary operation

```cpp
tensor.apply(MyUnaryFunction{});
```

The function receives a writable element:

```cpp
template <class T>
struct SetZero {
    TZ_HOST_DEVICE void operator()(T& value) const {
        value = T(0);
    }
};
```

### Binary operation

```cpp
destination.apply(source, MyBinaryFunction{});
```

The function receives a read-only source and writable destination:

```cpp
template <class T>
struct Square {
    TZ_HOST_DEVICE void operator()(const T& source, T& destination) const {
        destination = source * source;
    }
};
```

### Ternary operation

```cpp
destination.apply(a, b, MyTernaryFunction{});
```

The function receives two read-only inputs and one writable output:

```cpp
template <class T>
struct Add {
    TZ_HOST_DEVICE void operator()(
        const T& a,
        const T& b,
        T& output
    ) const {
        output = a + b;
    }
};
```

GPU-capable operations must pass `tz::impl::AnyDevice{}`:

```cpp
tensor.apply(MyFunction{}, tz::impl::AnyDevice{});
```

Without `AnyDevice`, the operation is treated as CPU-only.

## Custom GPU Functions

Place custom CUDA-compatible functors in `include/ExtraGPUFunctions.hpp`.

Functions intended for both CPU and GPU must use `TZ_HOST_DEVICE`:

```cpp
template <class T>
struct Square {
    TZ_HOST_DEVICE void operator()(const T& source, T& destination) const {
        destination = source * source;
    }
};
```

Instantiate the required CUDA kernel:

```cpp
#define TZ_EXTRA_GPU_FUNCTIONS_INSTANTIATE(T) \
    INSTANTIATE_BINARY_APPLY(T, Square);
```

Available macros:

```text
INSTANTIATE_UNARY_APPLY(T, Func)
INSTANTIATE_BINARY_APPLY(T, Func)
INSTANTIATE_TERNARY_APPLY(T, Func)
```

## Allocators

All allocators implement the `tz::mem::Allocator` interface.

| Allocator         | Device | Description                               |
| ----------------- | ------ | ----------------------------------------- |
| `tz::mem::Salloc` | CPU    | Thread-local three-tier slab allocator    |
| `tz::mem::Malloc` | CPU    | Wrapper around `aligned_alloc` and `free` |
| `tz::mem::Galloc` | GPU    | Wrapper around CUDA allocation functions  |

### Salloc tiers

`Salloc` dispatches allocations by size:

| Allocation size | Allocator tier   |
| --------------- | ---------------- |
| Up to 4 KiB     | Small allocator  |
| 4 KiB to 1 MiB  | Medium allocator |
| Above 1 MiB     | Large allocator  |

The default allocator is selected by device:

```cpp
if (device == CPU)
    return Salloc::instance();

if (device == GPU)
    return Galloc::instance();

return Malloc::instance();
```

## Benchmarks

The benchmark executable covers:

- Small, medium, and large allocations
- Tensor creation and destruction
- Large CPU and GPU element-wise addition
- CPU and GPU matrix multiplication
- CPU and GPU autograd training steps

Build the release benchmark first (it it important to enable tests):

```sh
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DTESTS=ON

cmake --build build -j"$(nproc)"
```

Run one benchmark pass directly:

```sh
./build/bench
```

Run repeated benchmarks and save the median result under the current Git hash:

```sh
python3 bench.py <number of benchmarks>
```

`bench.py` reports the median over all runs. The displayed percentage is the sample standard deviation relative to that median.

### Recorded results

The following results are from the recorded `b9ea905` run using 17 complete benchmark passes.

#### Allocator

| Benchmark                                     |  Median | Variation |
| --------------------------------------------- | ------: | --------: |
| Small `Salloc`, 1 B–4 KiB                     | 29.3 ns |     ±1.0% |
| Small `Malloc`, 1 B–4 KiB                     | 72.8 ns |     ±1.4% |
| Medium `Salloc`, 8–512 KiB                    | 21.3 ns |     ±1.0% |
| Medium `Malloc`, 8–512 KiB                    |  1.9 µs |     ±1.4% |
| Large `Salloc`, 2–16 MiB                      | 52.3 ns |     ±5.1% |
| Large `Malloc`, 2–16 MiB                      |  3.7 µs |     ±1.2% |
| `Vector<float>(256)` creation and destruction | 67.1 ns |     ±3.0% |

These tests repeatedly allocate a batch of randomly sized blocks, touch the last byte of each allocation, and then deallocate the batch. `Salloc` benefits from retaining and reusing previously allocated slabs.

#### Element-wise addition

The element-wise benchmark processes approximately 16 million floats per vector:

| Device | Median | Variation |
| ------ | -----: | --------: |
| CPU    | 9.3 ms |     ±0.3% |
| GPU    | 1.3 ms |     ±1.3% |

The GPU result includes synchronization and an amortized device-to-host copy used to consume the result.

#### Matrix multiplication

| Benchmark         |   Median |     Throughput |
| ----------------- | -------: | -------------: |
| CPU `256 × 256`   |   3.6 ms |    9.4 GFLOP/s |
| CPU `512 × 512`   |  27.5 ms |    9.8 GFLOP/s |
| GPU `256 × 256`   |  19.5 µs | 1718.1 GFLOP/s |
| GPU `512 × 512`   | 165.8 µs | 1619.5 GFLOP/s |
| GPU `1024 × 1024` | 646.1 µs | 3323.8 GFLOP/s |

CPU matrix multiplication uses the TenZor/OpenMP implementation. GPU `float` matrix multiplication uses cuBLAS.

#### Autograd training step

The current recorded training benchmark uses a small `2 → 8 → 8 → 1` network with sigmoid activations and SGD:

| Device | Median per step | Variation |
| ------ | --------------: | --------: |
| CPU    |         79.3 µs |     ±0.5% |
| GPU    |        708.5 µs |     ±1.8% |

For a network this small, GPU kernel-launch and synchronization overhead dominates the computation, making the CPU faster. Larger training and transformer-oriented benchmarks are planned as the benchmark suite grows.

## Configuration

Configuration values can be supplied as compile-time definitions or defined before including `TenZor.hpp`.

| Define                             | Default            | Description                                                             |
| ---------------------------------- | ------------------ | ----------------------------------------------------------------------- |
| `TZ_CUDA_AVAILABLE`                | Build-defined      | Whether CUDA support is available                                       |
| `TZ_MAX_DIM`                       | `4`                | Maximum tensor dimension                                                |
| `TZ_START_MEM_SIZE`                | `10 * 1024 * 1024` | Initial `Salloc` pool size                                              |
| `TZ_DEFAULT_ALIGNMENT`             | `64`               | Default allocation alignment                                            |
| `TZ_CTD`                           | `1`                | Enable cross-thread deallocation under the retirement restriction below |
| `TZ_ERRORS`                        | `1`                | Enable runtime validation and exceptions                                |
| `TZ_IMMUTABLE_BROADCASTS`          | `1`                | Prevent writes through broadcasted views                                |
| `TZ_APPLY_ERROR_IF_SHAPE_NOT_SAME` | `1`                | Report shape mismatches in `apply`                                      |
| `TZ_DEFAULT_ALLOCATOR`             | Device-dependent   | Override default allocator selection                                    |

Example:

```sh
-DTZ_MAX_DIM=8
```

Or before including TenZor:

```cpp
#define TZ_MAX_DIM 8
#include "TenZor.hpp"
```

## Known Limitations

- Autograd graphs are single-use and are consumed by `backward()`.
- Differentiable operands and intermediates must remain alive until backward completes.
- CUDA `matmul` is implemented only for `float` and `double`.
- Host access through `at()` and `get()` is unavailable for GPU tensors.
- `Salloc` supports alignments up to 64 bytes.
- Objects backed by the thread-local allocator should not use static storage duration.
- Nested `std::vector` and initializer-list tensor inputs must be rectangular.
- Cross-thread deallocation is supported only after the allocating thread has retired and will never resume or access its allocator again. Concurrent allocator access across those threads is unsupported.
