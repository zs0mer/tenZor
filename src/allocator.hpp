#pragma once
#include "../src/tenzor_utils.hpp"
//^ not needed, will remove this

namespace TZ {
namespace mem {

enum Device { CPU, CUDA };

// an abstract class
class Allocator {
  public:
	virtual Device device() const = 0;

	virtual void* allocate(const size_t bytes, const size_t alignment) = 0;

	virtual void* deallocate(void* ptr, const size_t bytes) = 0;

	virtual ~Allocator() = default;
};

class SmallAllocator;
class MediumAllocator;
class LargeAllocator;

//~     - 4KB (end is inclusive)
class SmallAllocator {
  private:
	//~ 16KB
	const inline static constexpr uint16_t SLABSIZE = 16 * 1024;
	const inline static constexpr uint16_t POOLTYPENUMBER = 10;
	const inline static constexpr uint32_t POOLSIZE[POOLTYPENUMBER] = // the possible bite pools
	    {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
	const inline static constexpr uint16_t POOLWEIGHT[POOLTYPENUMBER] = // weights for distributing
	    {10, 13, 15, 15, 13, 10, 8, 8, 4, 4};                           // the memory when refilling
	                                                                    // (adds up to 100)

	struct SmallSlab {
		void* blocks;
		uint16_t size;
		uint16_t blockSize;
		uint16_t freeBlocks;
	};

	struct FreeBlock {
		void* next;
	};

	//~ 1MB memory
	struct ThreadLocalCache {
		FreeBlock* bin[POOLTYPENUMBER] = {nullptr};
		uint16_t blockNum[POOLTYPENUMBER] = {0};
		const static constexpr uint16_t MAX_PER_BIN = 32;
	};

	thread_local static ThreadLocalCache tlc_;
	FreeBlock* globalBin_[POOLTYPENUMBER] = {nullptr};
	uint16_t blockNum_[POOLTYPENUMBER] = {0};
	MediumAllocator& midAlloc_;

  public:
	SmallAllocator(const uint32_t startPoolSize, MediumAllocator& midAlloc) : midAlloc_(midAlloc) {
		fillPool(startPoolSize);
		fillTLC();
	}

  private:
	void fillTLC() {
		for (int i = 0; i < POOLTYPENUMBER; i++)
			refillTLC(i);
	}

	void fillTLC(const uint16_t sizeType) {
		//* can be much faster
		for (int i = 0; i < tlc_.MAX_PER_BIN; ++i) {
			FreeBlock* block = globalBin_[sizeType];
			if (!block)
				break;
			globalBin_[sizeType] = (FreeBlock*)block->next;
			block->next = tlc_.bin[sizeType];
			tlc_.bin[sizeType] = block;
			tlc_.blockNum[sizeType]++;
		}
	}

	void fillPool(const uint32_t bites) {
		for (int i = 0; i < POOLTYPENUMBER; i++)
			fillPool((bites * POOLWEIGHT[i]) / 100, i);
	}

	void fillPool(uint32_t bites, const uint16_t sizeType) {
		//* can be much faster
		// if the allocated space is to small for a slab round it up to 1
		uint32_t numSlabs = (bites + SLABSIZE - 1) / SLABSIZE;


		for (int i = 0; i < numSlabs; i++) {
			const uint32_t blockSize = POOLSIZE[sizeType];

			void* slab = midAlloc_.alloc(SLABSIZE, 64);
			CHECK(!slab, "out of memory");

			uint8_t* ptr = (uint8_t*)slab;
			uint8_t* end = ptr + SLABSIZE;

			while (ptr + blockSize <= end) {
				FreeBlock* block = (FreeBlock*)ptr;

				block->next = globalBin_[sizeType];
				globalBin_[sizeType] = block;

				blockNum_[sizeType]++;

				ptr += blockSize;
			}
		}
	}
};

//~ 4KB - 1Mb (end is inclusive)
class MediumAllocator {
  private:
	const static constexpr uint32_t SLABSIZE = 4 * 1024 * 1024;

	struct MediumSlab {
		uint8_t* start;
		uint8_t* currentFree;
		size_t size;
	};

	std::vector<MediumSlab*> slabs;
	uint16_t activeSlab; // index pointing to the next active slab

  public:
	MediumAllocator(const uint32_t startPoolSize) {}

	void* alloc(const size_t bytes, const size_t alignment) {}
};

//~ 1MB -
class LargeAllocator {
  private:
	struct LargeBlock {
		size_t size;
		LargeBlock* next;
	};

	LargeBlock* block;

  public:
	LargeAllocator(const uint32_t startPoolSize) {}
};


// the standard CPU allocater
// do not does the "dirty" work (allocating)
// splits the work in to small, medium, large
class salloc : Allocator {
  private:
	const inline static constexpr uint16_t
	    initRatio[3] = // percentiges of the allocators (adds up to 100)
	    {70, 30, 0};

	SmallAllocator sa_;  //~        < 4KB (end is inclusive)
	MediumAllocator ma_; //~ 4KB <  < 1Mb (end is inclusive)
	LargeAllocator la_;  //~ 1MB >


  public:
	salloc(const uint32_t bitesToPool)
	    : sa_(bitesToPool * initRatio[0]), ma_(bitesToPool * initRatio[1]),
	      la_(bitesToPool * initRatio[2]) {}


	Device device() {
		return Device::CPU;
	};

	void* allocate(const size_t bytes, const size_t alignment = 64) {};

	void* deallocate(void* ptr, const size_t bytes) {};

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