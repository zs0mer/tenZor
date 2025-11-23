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

//~    <- 4KB (end is inclusive)
class SmallAllocator {
  private:
	//~ 16KB
	const static constexpr uint16_t REFILLSIZE = 2;
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
		if (startPoolSize == 0)
			return;

		fillPool(startPoolSize);
		fillTLC();
	}

	inline void* alloc(const size_t bytes) {

		uint16_t sizeType = POOLTYPENUMBER;
		for (uint16_t i = 0; i < POOLTYPENUMBER; ++i) {
			if (bytes <= POOLSIZE[i]) {
				sizeType = i;
				break;
			}
		}


		CHECK_(sizeType == POOLTYPENUMBER);

		FreeBlock* block = tlc_.bin[sizeType];
		if (block) [[likely]] {
			tlc_.bin[sizeType] = static_cast<FreeBlock*>(block->next);
			tlc_.blockNum[sizeType]--;
			return static_cast<void*>(block);
		}


		fillTLC(sizeType);


		if (tlc_.bin[sizeType]) {
			fillPool(REFILLSIZE, sizeType);
			fillTLC(sizeType);
		}

		return alloc(bytes);
	}

  private:
	inline void fillTLC() {
		for (uint32_t i = 0; i < POOLTYPENUMBER; i++)
			fillTLC(i);
	}

	inline void fillTLC(const uint16_t sizeType) {
		//* can be much faster
		for (uint32_t i = 0; i < tlc_.MAX_PER_BIN; ++i) {
			FreeBlock* block = globalBin_[sizeType];
			if (!block)
				break;
			globalBin_[sizeType] = static_cast<FreeBlock*>(block->next);
			block->next = tlc_.bin[sizeType];
			tlc_.bin[sizeType] = block;
			tlc_.blockNum[sizeType]++;
		}
	}

	inline void fillPool(const uint32_t bites) {
		for (uint32_t i = 0; i < POOLTYPENUMBER; i++)
			fillPool((bites * POOLWEIGHT[i]) / 100, i);
	}

	inline void fillPool(uint32_t bites, const uint16_t sizeType) {
		//* can be much faster
		// if the allocated space is to small for a slab round it up to 1
		uint32_t numSlabs = (bites + SLABSIZE - 1) / SLABSIZE;


		for (uint32_t i = 0; i < numSlabs; i++) {
			const uint32_t blockSize = POOLSIZE[sizeType];

			void* slab = midAlloc_.alloc(SLABSIZE, 64);
			CHECK(!slab, "out of memory");

			void* ptr = static_cast<void*>(slab);
			void* end = ptr + SLABSIZE;

			while (ptr + blockSize <= end) {
				FreeBlock* block = static_cast<FreeBlock*>(ptr);

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
	const static constexpr uint16_t REFILLSIZE = 2;

	struct MediumSlab {
		uint8_t* start;
		uint8_t* currentFree;
		size_t size;
	};

	std::vector<MediumSlab*> slabs;
	uint16_t activeSlab = 0; // index pointing to the next active slab

  public:
	MediumAllocator(const uint32_t startPoolSize) {
		if (startPoolSize == 0)
			return;

		uint16_t slabNum = (startPoolSize + SLABSIZE - 1) / SLABSIZE;

		fillSlabs(slabNum);

		activeSlab = 0;
	}


	inline void* alloc(const size_t bytes, const size_t alignment) {
		if (slabs.size() <= activeSlab)
			fillSlabs(REFILLSIZE);


		MediumSlab* slab = slabs[activeSlab];

		// align the current pointer
		// some wizard magic
		uintptr_t currentAddr = reinterpret_cast<uintptr_t>(slab->currentFree);
		uintptr_t alignedAddr = (currentAddr + alignment - 1) & ~(alignment - 1);


		if (alignedAddr + bytes <= reinterpret_cast<uintptr_t>(slab->start + slab->size))
		    [[likely]] {

			slab->currentFree = reinterpret_cast<uint8_t*>(alignedAddr + bytes);
			void* ptr = reinterpret_cast<void*>(alignedAddr);
			return ptr;
		}

		activeSlab++;
		if (slabs.size() <= activeSlab) [[likely]]
			fillSlabs(REFILLSIZE);

		return alloc(bytes, alignment);
	}

  private:
	inline void fillSlabs(uint16_t slabNum) {
		for (uint32_t i = 0; i < slabNum; ++i) {
			uint8_t* mem = static_cast<uint8_t*>(std::aligned_alloc(64, SLABSIZE));
			CHECK_(!mem);

			MediumSlab* slab = new MediumSlab{.start = mem, .currentFree = mem, .size = SLABSIZE};

			slabs.push_back(slab);
		}
	}
};

//~ 1MB ->
class LargeAllocator {
  private:
	struct LargeBlock {
		size_t size;
		uint16_t alignment;
		LargeBlock* next;
	};

	LargeBlock* block = nullptr;

  public:
	LargeAllocator(const uint32_t startPoolSize) {
		CHECK(startPoolSize != 0, "can't allocate pool in large allocator");
	}

	void* alloc(const size_t bytes, const size_t alignment) {
		LargeBlock* ptr = block;
		LargeBlock* last = nullptr;

		while (ptr) {
			uintptr_t raw = reinterpret_cast<uintptr_t>(ptr + 1); // +1 to keep the header
			uintptr_t aligned = (raw + alignment - 1) & ~(alignment - 1);

			if (aligned + bytes < raw + ptr->size) {
				if (last)
					last->next = ptr->next;
				else
					block = ptr->next;

				return reinterpret_cast<void*>(aligned);
			}
			last = ptr;
			ptr = ptr->next;
		}

		uint32_t size = ((bytes + alignment - 1) / alignment) * alignment;

		return std::aligned_alloc(alignment, size);
	}
};


// the standard CPU allocater
// do not does the "dirty" work (allocating)
// splits the work in to small, medium, large
class salloc : Allocator {
  private:
	// percentiges of the allocators
	// (adds up to 100)
	const inline static constexpr uint16_t initRatio[3] = {50, 50, 0};

	SmallAllocator sa_;  //~        < 4KB (end is inclusive)
	MediumAllocator ma_; //~ 4KB <  < 1Mb (end is inclusive)
	LargeAllocator la_;  //~ 1MB >


  public:
	salloc(const uint32_t bitesToPool)
	    : ma_(bitesToPool * initRatio[1]), sa_(bitesToPool * initRatio[0], ma_),
	      la_(bitesToPool * initRatio[2]) {}


	inline Device device() {
		return Device::CPU;
	};

	inline void* allocate(const size_t bytes, const size_t alignment = 64) {
		if (bytes <= 4 * 1024) {                //~ 0b
			return sa_.alloc(bytes);            //~
		} else if (bytes <= 1024 * 1024) {      //~ 4KB
			return ma_.alloc(bytes, alignment); //~
		} else {                                //~ 1Mb
			return la_.alloc(bytes, alignment); //~
		}
	};

	inline void* deallocate(void* ptr, const size_t bytes) {};

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