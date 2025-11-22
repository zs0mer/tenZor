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

class MediumAllocator;

//     - 4KB (end is inclusive)
class SmallAllocator {
  private:
    const static constexpr uint16_t TOTALPOOLSIZE = 10;
    const static constexpr uint32_t POOLSIZE[TOTALPOOLSIZE] = // the possible bite pools
        {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    struct SmallSlab {
        void* blocks;
        uint16_t size;
        uint16_t blockSize;
        uint16_t freeBlocks;
    };

    struct FreeBlock {
        void* next;
    };

    struct SizeClass {
        uint32_t blockSize;
        FreeBlock* blocks;
        std::vector<SmallSlab*> usedSlabs;
    };

    struct ThreadLocalCache {
        FreeBlock* freeLists[TOTALPOOLSIZE] = {nullptr};
        int usable[TOTALPOOLSIZE] = {0};
    };

    thread_local static ThreadLocalCache tlc_;

    std::vector<SizeClass> sizeType_;
    MediumAllocator& midAlloc_;
};

// 4KB - 1Mb (end is inclusive)
class MediumAllocator {
  private:
    const static constexpr uint32_t SLABSIZE = 4 * 1024 * 1024;

    struct MediumSlab {
        uint8_t* start;
        uint8_t* currentFree;
        size_t size;
    };

    std::vector<MediumSlab*> slabs;
    uint16_t activeSlab;
};

// 1MB -
class LargeAllocator {
  private:
    struct LargeBlock {
        size_t size;
        LargeBlock* next;
    };
};


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