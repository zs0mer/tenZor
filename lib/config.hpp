#pragma once
#include <cstdint>

namespace TZ {

namespace mem {

class Allocator;
class LargeAllocator;
class MediumAllocator;
class SmallAllocator;
class Salloc;
class Malloc;

class BufferIMPL;
class Buffer;
} // namespace mem

template <class T>
class Tensor;
template <typename Derived, typename T>
class _tensorWrapper;

template <typename T>
class Scalar;
template <typename T>
class Vector;
template <typename T>
class Matrix;

} // namespace TZ

#ifdef TENZOR_USER_CONFIG
#include TENZOR_USER_CONFIG
#else

#define TZ_ERRORS 1
#define TZ_NORMAL_EQUAL 1

namespace TZ::config {

constexpr std::uint8_t MAX_DIM = 64;
constexpr std::uint64_t DEFAULT_ALIGNMENT = 64;
constexpr std::uint64_t START_MEM_SIZE = 10 * 1024 * 1024;
inline TZ::mem::Allocator& defaultAllocator();

} // namespace TZ::config

#endif
