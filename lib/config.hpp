#pragma once


#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <mutex>
#include <cstring>
#include <array>
#include <span>
#include <iostream>
#include <functional>

namespace TZ::mem {
class Allocator;
class BufferIMPL;
class Buffer;
} // namespace TZ::mem

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