#pragma once

namespace TZ {
namespace mem {

enum Device { CPU, CUDA };

// an abstract class
class Allocator {
  public:
    virtual Device device() const = 0;

    virtual void* allocate(size_t bytes, size_t alignment = 1) = 0;

    virtual void* deallocate(void* ptr, size_t bytes) = 0;

    virtual ~Allocator() = default;
};

class SmallAllocator {};

class MedAllocator {};

class LargeAllocator {};

// the standard CPU allocater
// do not does the "dirty" work (allocating)
// splits the work in to small, medium, large
class salloc : Allocator {
  private:
    SmallAllocator sa;
    MedAllocator ma;
    LargeAllocator la;

  public:
    virtual Device device() {
        return Device::CPU;
    };

    virtual void* allocate(size_t bytes, size_t alignment = 1) = 0;

    virtual void* deallocate(void* ptr, size_t bytes) = 0;

    virtual ~salloc() = default;
};


class Buffer {
  private:
    void* const data;
    const size_t sizeBites;
    const uint32_t alignment;
    Allocator* allocator;
    std::atomic<uint32_t> refCount{1};

  public:
    Buffer(const size_t sizeBites, const uint32_t alignment, Allocator* allocator)
        : sizeBites(sizeBites), alignment(alignment), allocator(allocator),
          data(allocator->allocate(sizeBites, alignment)) {}

    ~Buffer() {
        allocator->deallocate(data, sizeBites);
    }
};

} // namespace mem
} // namespace TZ