#pragma once

#include <cstdint>
#include <mutex>
#include <cstdlib>
#include <vector>
#include <atomic>

#include "kernel_functions.hpp"
#include "utils.hpp"

namespace tz {

enum Device { CPU, GPU };

namespace mem {

// # -------------------------
#ifdef TZ_START_MEM_SIZE
const constexpr std::uint64_t START_MEM_SIZE = TZ_START_MEM_SIZE;
#else
const constexpr std::uint64_t START_MEM_SIZE = 10 * 1024 * 1024;
#endif
// # -------------------------
#ifdef TZ_DEFAULT_ALIGNMENT
const constexpr std::uint8_t DEFAULT_ALIGNMENT = TZ_DEFAULT_ALIGNMENT;
#else
const constexpr std::uint8_t DEFAULT_ALIGNMENT = 64;
#endif
// # -------------------------
#ifdef TZ_CTD
// nothing
#else
#define TZ_CTD 1
#endif
// # -------------------------

// an abstract class
// this declears an interface for allocating
class Allocator {
  public:
	virtual Device device() const = 0;

	virtual void* allocate(const uint64_t bytes, const uint8_t alignment) = 0;

	virtual void deallocate(void* ptr, const uint64_t bytes) = 0;

	virtual ~Allocator() = default;

	Allocator() = default;


	Allocator(const Allocator&) = delete;

	Allocator& operator=(const Allocator&) = delete;

	Allocator(Allocator&&) = delete;

	Allocator& operator=(Allocator&&) = delete;
};

// # ================================================================================

// this is an allocator
// can allocate bytes in the range of (1MB; INF)
// uses headers
class LargeAllocator {
  private:
	struct LargeBlock {
		uint64_t size = 0;
		LargeBlock* next = nullptr;
	};

	std::mutex mtx_;
	LargeBlock* blocks_ = nullptr;

  public:
	void* alloc(const uint64_t bytes, const uint16_t alignment) {
		std::lock_guard<std::mutex> lock(mtx_);
		LargeBlock* ptr = blocks_;
		LargeBlock* last = nullptr;

		while (ptr) {
			if (bytes <= ptr->size) {
				if (last)
					last->next = ptr->next;
				else
					blocks_ = ptr->next;

				return ptr;
			}
			last = ptr;
			ptr = ptr->next;
		}

		uint64_t size = ((bytes + DEFAULT_ALIGNMENT - 1) / DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT;

		return std::aligned_alloc(DEFAULT_ALIGNMENT, size);
	}

	void dealloc(void* ptr, const uint64_t bytes) {
		std::lock_guard<std::mutex> lock(mtx_);
		LargeBlock* currBlock = reinterpret_cast<LargeBlock*>(ptr);
		uint64_t size = ((bytes + DEFAULT_ALIGNMENT - 1) / DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT;

		currBlock->size = size;
		currBlock->next = blocks_;
		blocks_ = currBlock;
	}

	~LargeAllocator() {
		LargeBlock* block = blocks_;
		while (block) {
			LargeBlock* next = block->next;
			free(block);
			block = next;
		}
	}

  private:
	friend class Salloc;

	LargeAllocator() = default;
};

// # ================================================================================

// this is an allocator
// can allocate bytes in the range of (4KB; 1MB]
// uses masking
class MediumAllocator {
  private:
	struct MediumSlab {
		uint8_t* freeMem = nullptr;
		std::atomic<uint32_t> allocatedBlocks{0};
		uint64_t idxInBin = 0;
	};

	const static uintptr_t SLABSIZE = 4 * 1024 * 1024; // ! must be a power of two
	const static uint16_t REFILLSIZE = 2;
	const static uint16_t SLABHEADERSIZE = 64;

	std::vector<MediumSlab*> bin_;
	uint16_t activeSlabIdx_ = 0;

  public:
	MediumAllocator() = delete;

	void* alloc(const uint64_t bytes, const uint16_t alignment) {
		while (true) {
			if (bin_.size() <= activeSlabIdx_)
				fillSlabs(REFILLSIZE);


			MediumSlab* slab = bin_[activeSlabIdx_];


			uintptr_t currentAddr = reinterpret_cast<uintptr_t>(slab->freeMem);
			uintptr_t alignedAddr = (currentAddr + alignment - 1) & ~(alignment - 1);
			uintptr_t slabEnd = reinterpret_cast<uintptr_t>(slab) + SLABSIZE;

			if (alignedAddr + bytes <= slabEnd) [[likely]] {

				slab->freeMem = reinterpret_cast<uint8_t*>(alignedAddr + bytes);
				void* ptr = reinterpret_cast<void*>(alignedAddr);
				slab->allocatedBlocks.fetch_add(1, std::memory_order_relaxed);
				return ptr;
			}
			activeSlabIdx_++;
		}

		return nullptr;
	}

	void dealloc(void* ptr) {
		MediumSlab* slab =
		    reinterpret_cast<MediumSlab*>(reinterpret_cast<uintptr_t>(ptr) & (~(SLABSIZE - 1)));

		if (slab->allocatedBlocks.fetch_sub(1, std::memory_order_acq_rel) != 1)
			return;

		slab->freeMem = reinterpret_cast<uint8_t*>(slab) + SLABHEADERSIZE;

		if (bin_.size() <= slab->idxInBin || bin_[slab->idxInBin] != slab) {
			// ! cross concurrent thread deallocation is not supported
			// ! but this is not safe IF the two threads are concurrent AT THE SAME TIME

#if TZ_CTD
			// the slab should not be available only for this thread
			slab->idxInBin = bin_.size();
			bin_.push_back(slab);
			return;
#else
			TZ_CHECK(false, "invalid pointer passed to MediumAllocator::dealloc, \ncross thread "
			                "deallocation is not supported");
#endif
		}

		if (activeSlabIdx_ == 0)
			return;

		std::swap(bin_[activeSlabIdx_ - 1], bin_[slab->idxInBin]);

		bin_[slab->idxInBin]->idxInBin = slab->idxInBin;
		bin_[activeSlabIdx_ - 1]->idxInBin = activeSlabIdx_ - 1;

		activeSlabIdx_--;
	}

	~MediumAllocator() {
		for (auto* i : bin_) {
			if (i->allocatedBlocks.load(std::memory_order_relaxed) == 0)
				free(i);
		}
	}

  private:
	MediumAllocator(const uint32_t startPoolSize) {
		if (startPoolSize == 0)
			return;

		uint16_t slabNum = (startPoolSize + SLABSIZE - 1) / SLABSIZE;

		fillSlabs(slabNum);
	}

	void fillSlabs(const uint16_t slabNum) {
		bin_.reserve(bin_.size() + slabNum);
		for (uint32_t i = 0; i < slabNum; i++) {
			MediumSlab* mem = static_cast<MediumSlab*>(std::aligned_alloc(SLABSIZE, SLABSIZE));
			TZ_CHECK_(mem);

			mem->freeMem = reinterpret_cast<uint8_t*>(mem) + SLABHEADERSIZE;
			mem->allocatedBlocks.store(0, std::memory_order_relaxed);
			mem->idxInBin = bin_.size();

			bin_.push_back(mem);
		}
	}

	friend class Salloc;
};

// # ================================================================================

// this is an allocator
// can allocate bytes in the range of (0; 4KB]
// uses masking
class SmallAllocator {
  private:
	struct FreeBlock {
		FreeBlock* next = nullptr;
	};

	// * Treiber stack
	struct BlockStack {
		std::atomic<FreeBlock*> head{nullptr};

		void push(FreeBlock* block) {
			block->next = head.load(std::memory_order_relaxed);
			while (!head.compare_exchange_weak(block->next, block, std::memory_order_release,
			                                   std::memory_order_relaxed)) {
				block->next = head.load(std::memory_order_relaxed);
			}
		}

		FreeBlock* pop() {
			FreeBlock* block = head.load(std::memory_order_acquire);
			while (block &&
			       !head.compare_exchange_weak(block, block->next, std::memory_order_acquire,
			                                   std::memory_order_relaxed)) {
			}
			return block;
		}
	};

	struct SmallSlab {
		uint16_t blockSizeType = UINT16_MAX;
		SmallSlab* nextSlab = nullptr;
		BlockStack nextFreeBlock;
		std::atomic<uint32_t> allocatedBlocks{0};
		std::atomic<bool> available{true};
	};


	const static uint16_t REFILLSIZE = 2;          // 16KB
	const static uint16_t SLABHEADERSIZE = 64 * 4; // ! this is the size to preserve max alignment
	const static uintptr_t SLABSIZE = 32 * 1024;   // ! must be a power of two
	const static uint16_t POOLTYPENUMBER = 10;
	const static constexpr uint32_t POOLSIZE[POOLTYPENUMBER] = // the possible bite pools
	    {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
	const static constexpr uint16_t POOLWEIGHT[POOLTYPENUMBER] = // weights for distributing
	    {6, 8, 10, 10, 10, 10, 10, 10, 12, 14};                  // the memory when creating
	                                                             // * adds up to 100

	SmallSlab* bin_[POOLTYPENUMBER] = {nullptr};
	MediumAllocator& midAlloc_;

  public:
	SmallAllocator() = delete;

	void* alloc(const uint64_t bytes) {
		// determining the sizeType
		uint16_t sizeType = POOLTYPENUMBER;
		for (uint16_t i = 0; i < POOLTYPENUMBER; i++) {
			if (bytes <= POOLSIZE[i]) {
				sizeType = i;
				break;
			}
		}

		TZ_CHECK_(sizeType < POOLTYPENUMBER);

		while (true) {
			SmallSlab* slab = bin_[sizeType];
			if (slab) [[likely]] {
				TZ_CHECK_(slab->nextFreeBlock.head.load(std::memory_order_relaxed));

				void* block = slab->nextFreeBlock.pop();
				slab->allocatedBlocks.fetch_add(1, std::memory_order_relaxed);
				if (!slab->nextFreeBlock.head.load(std::memory_order_relaxed)) {
					slab->available.store(false, std::memory_order_release);
					bin_[sizeType] = slab->nextSlab;
				}

				return static_cast<void*>(block);
			}
			fillPool(REFILLSIZE * SLABSIZE, sizeType);
		}
		return nullptr;
	}

	void dealloc(void* ptr) {
		SmallSlab* slab =
		    reinterpret_cast<SmallSlab*>(reinterpret_cast<uintptr_t>(ptr) & (~(SLABSIZE - 1)));

		FreeBlock* block = reinterpret_cast<FreeBlock*>(ptr);
		slab->nextFreeBlock.push(block);

		if (slab->allocatedBlocks.fetch_sub(1, std::memory_order_acq_rel) != 1 ||
		    slab->available.load(std::memory_order_acquire))
			return;

		slab->available.store(true, std::memory_order_relaxed);
		slab->nextSlab = bin_[slab->blockSizeType];
		bin_[slab->blockSizeType] = slab;
	}

	~SmallAllocator() {
		for (uint16_t t = 0; t < POOLTYPENUMBER; t++) {
			SmallSlab* slab = bin_[t];
			while (slab) {
				SmallSlab* next = slab->nextSlab;
				if (slab->allocatedBlocks.load(std::memory_order_relaxed) == 0)
					midAlloc_.dealloc(slab);
				else
					slab->available.store(false, std::memory_order_relaxed);
				slab = next;
			}
		}
	}

  private:
	SmallAllocator(const uint32_t startPoolSize, MediumAllocator& midAlloc) : midAlloc_(midAlloc) {
		if (startPoolSize == 0)
			return;

		fillPool(startPoolSize);
	}

	void fillPool(const uint32_t bytes) {
		for (uint32_t i = 0; i < POOLTYPENUMBER; i++)
			fillPool((bytes * POOLWEIGHT[i]) / 100, i);
	}

	void fillPool(const uint32_t bytes, const uint16_t sizeType) {
		// * can be faster
		// if the allocated space is to small for a slab round it up to 1
		uint32_t numSlabs = (bytes + SLABSIZE - 1) / SLABSIZE;


		for (uint32_t i = 0; i < numSlabs; i++) {
			const uint32_t blockSize = std::max<uint32_t>(POOLSIZE[sizeType], sizeof(FreeBlock));
			SmallSlab* slab = static_cast<SmallSlab*>(midAlloc_.alloc(SLABSIZE, SLABSIZE));

			TZ_CHECK(slab, "out of memory");

			new (slab) SmallSlab();

			slab->nextSlab = bin_[sizeType];
			slab->blockSizeType = sizeType;
			bin_[sizeType] = slab;

			uint8_t* ptr = reinterpret_cast<uint8_t*>(slab) + SLABHEADERSIZE;
			uint8_t* end = ptr + SLABSIZE - SLABHEADERSIZE;

			while (ptr + blockSize <= end) {
				reinterpret_cast<FreeBlock*>(ptr)->next =
				    slab->nextFreeBlock.head.load(std::memory_order_relaxed);
				slab->nextFreeBlock.push(reinterpret_cast<FreeBlock*>(ptr));

				ptr = ptr + blockSize;
			}
		}
	}

	friend class Salloc;
};

// # ================================================================================

// a CPU allocator
// ! singleton
// max alignment: 64
// alignment can only be 2^n
class Salloc : public Allocator {
  private:
	// percentiges of the allocators (small, medium)
	// ! has to add up to 100%
	const static constexpr uint16_t INITRATIO[2] = {50, 50};

	LargeAllocator la_;

	MediumAllocator& ma_() {
		static thread_local MediumAllocator ma_(START_MEM_SIZE * INITRATIO[1] / 100);
		return ma_;
	}

	SmallAllocator& sa_() {
		static thread_local SmallAllocator sa_(START_MEM_SIZE * INITRATIO[0] / 100, ma_());
		return sa_;
	}

	Salloc() = default;


  public:
	static Salloc& instance() {
		static Salloc* s = new Salloc;
		return *s;
	}

	Device device() const override {
		return Device::CPU;
	};

	// * if size < alignment, alignment will not be used
	// * alignment can be maximum 64 bytes
	// * alignment can only be powers of 2
	// * cross concurrent thread deallocation is not supported
	void* allocate(const uint64_t bytes, const uint8_t alignment) override {
		if (bytes == 0)
			return nullptr;
		TZ_CHECK(alignment <= 64, "alignment exceeds maximum of 64");
		if (bytes <= 4 * 1024) {                  //~ 0b
			return sa_().alloc(bytes);            //~
		} else if (bytes <= 1024 * 1024) {        //~ 4KB
			return ma_().alloc(bytes, alignment); //~
		} else {                                  //~ 1Mb
			return la_.alloc(bytes, alignment);   //~
		}
	};

	void deallocate(void* ptr, const uint64_t bytes) override {
		if (bytes <= 4 * 1024) {            //~ 0b
			return sa_().dealloc(ptr);      //~
		} else if (bytes <= 1024 * 1024) {  //~ 4KB
			return ma_().dealloc(ptr);      //~
		} else {                            //~ 1Mb
			return la_.dealloc(ptr, bytes); //~
		}
	};

	~Salloc() = default;


	Salloc(const Salloc&) = delete;

	Salloc& operator=(const Salloc&) = delete;

	Salloc(Salloc&&) = delete;

	Salloc& operator=(Salloc&&) = delete;
};

// # ================================================================================

// simple allocator using malloc() and free()
class Malloc : public Allocator {
  private:
	Malloc() = default;


  public:
	static Malloc& instance() {
		static Malloc* s = new Malloc;
		return *s;
	}

	Device device() const override {
		return Device::CPU;
	};

	void* allocate(const uint64_t bytes, const uint8_t alignment) override {
		uint64_t size = ((bytes + alignment - 1) / alignment) * alignment;

		return aligned_alloc(alignment, size);
	};

	void deallocate(void* ptr, const uint64_t bytes = 0) override {
		free(ptr);
	};

	~Malloc() = default;


	Malloc(const Malloc&) = delete;

	Malloc& operator=(const Malloc&) = delete;

	Malloc(Malloc&&) = delete;

	Malloc& operator=(Malloc&&) = delete;
};

// # ================================================================================

// simple allocator to the GPU
// uses simple CUDA functions
class Galloc : public Allocator {
  private:
	Galloc() = default;


  public:
	static Galloc& instance() {
		static Galloc* s = new Galloc;
		return *s;
	}

	Device device() const override {
		return Device::GPU;
	};

	void* allocate(const uint64_t bytes, const uint8_t alignment) override {
		return cuda::allocGPU(bytes, alignment);
	};

	void deallocate(void* ptr, const uint64_t bytes = 0) override {
		cuda::freeGPU(ptr, bytes);
	};

	~Galloc() = default;


	Galloc(const Galloc&) = delete;

	Galloc& operator=(const Galloc&) = delete;

	Galloc(Galloc&&) = delete;

	Galloc& operator=(Galloc&&) = delete;
};

// # ================================================================================

inline Allocator& defaultAllocator(Device device = CPU) {
	// # -------------------------
#ifdef TZ_DEFAULT_ALLOCATOR
	TZ_DEFAULT_ALLOCATOR;
#else
	if (device == CPU)
		return Salloc::instance();

	if (device == GPU)
		return Galloc::instance();

	return Malloc::instance();
#endif
	// # -------------------------
}

} // namespace mem
} // namespace tz
