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


//     - 4KB (end is inclusive)
class SmallAllocator {
  private:
    struct SmallSlab {
        void* start;
        uint16_t size;
        uint16_t blockSize;
        uint16_t freeBlocks;
    };

    struct FreeBlock {
        void* next;
    };

    struct SizeClass {
        uint32_t block_size;
        FreeBlock* free_list;
        std::vector<SmallSlab*> slabs;
    };

    std::vector<SizeClass> sizeType_;
    MediumAllocator& midAlloc_;
};

// 4KB - 1Mb (end is inclusive)
class MediumAllocator {};

// 1MB -
class LargeAllocator {};


// the standard CPU allocater
// do not does the "dirty" work (allocating)
// splits the work in to small, medium, large
class salloc : Allocator {
  private:
    SmallAllocator sa_;  //        < 4KB (end is inclusive)
    MediumAllocator ma_; // 4KB <  < 1Mb (end is inclusive)
    LargeAllocator la_;  // 1MB >

  public:
    Device device() {
        return Device::CPU;
    };

    void* allocate(size_t bytes, size_t alignment = 1) = 0;

    void* deallocate(void* ptr, size_t bytes) = 0;

    ~salloc() = default;
};

class Buffer {
  private:
    void* const data_;
    const size_t size_;
    const uint32_t alignment_;
    Allocator* allocator_;
    std::atomic<uint32_t> refCount_{1};

  public:
    Buffer(const size_t size, const uint32_t alignment, Allocator* allocator)
        : size_(size), alignment_(alignment), allocator_(allocator),
          data_(allocator->allocate(size, alignment)) {}

    ~Buffer() {
        allocator_->deallocate(data_, size_);
    }
};

} // namespace mem
} // namespace TZ