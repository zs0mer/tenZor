#pragma once

namespace TZ {
namespace mem {

enum Device { CPU, CUDA };

// an abstract class
class Allocator {
  public:
	virtual inline Device device() const = 0;

	virtual inline void* allocate(const size_t bytes, const size_t alignment) = 0;

	virtual inline void deallocate(void*& ptr, const size_t bytes) = 0;

	virtual ~Allocator() = default;
};

//& ================================================================================
//~ 1MB ->
class LargeAllocator {
  private:
	struct LargeBlock {
		size_t size;
		LargeBlock* next = nullptr;
	};

	LargeBlock* blocks = nullptr;

  public:
	LargeAllocator() {}

	inline void* alloc(const size_t bytes, const size_t alignment) {
		LargeBlock* ptr = blocks;
		LargeBlock* last = nullptr;

		while (ptr) {
			uintptr_t raw = reinterpret_cast<uintptr_t>(ptr + 1); // +1 to keep the header
			uintptr_t aligned = (raw + alignment - 1) & ~(alignment - 1);

			if (aligned + bytes <= raw + ptr->size) {
				if (last)
					last->next = ptr->next;
				else
					blocks = ptr->next;

				return reinterpret_cast<void*>(aligned);
			}
			last = ptr;
			ptr = ptr->next;
		}

		uint32_t size = ((bytes + alignment - 1) / alignment) * alignment;

		return std::aligned_alloc(alignment, size);
	}

	inline void dealloc(void* ptr, const size_t bytes) {
		LargeBlock* currBlock = static_cast<LargeBlock*>(ptr);
		currBlock->size = bytes;
		currBlock->next = blocks;
		blocks = currBlock;
	}

	~LargeAllocator() {
		while (blocks) {
			void* block = blocks;
			blocks = blocks->next;
			free(block);
		}
	}
};

//& ================================================================================
//~ 4KB - 1Mb (end is inclusive)
class MediumAllocator {
  private:
	struct MediumSlab {
		uint8_t* start;
		uint8_t* currentFree;
		size_t size;
		uint32_t allocatedBlocks;
		size_t indexInSlabs;
	};
	const static constexpr uintptr_t SLABSIZE = 4 * 1024 * 1024; //! must be a power of two
	const static constexpr uint16_t REFILLSIZE = 2;
	const static constexpr uint16_t SLABHEADERSIZE = sizeof(MediumSlab);


	std::vector<MediumSlab*> slabs;
	uint16_t activeSlab = 0; // index pointing to the next active slab

  public:
	MediumAllocator(const uint32_t startPoolSize) {
		if (startPoolSize == 0)
			return;

		uint16_t slabNum = (startPoolSize + SLABSIZE - 1) / SLABSIZE;

		fillSlabs(slabNum);
	}


	inline void* alloc(const size_t bytes, const size_t alignment) {
		if (slabs.size() <= activeSlab)
			fillSlabs(REFILLSIZE);


		MediumSlab* slab = slabs[activeSlab];

		// align the current pointer
		uintptr_t currentAddr = reinterpret_cast<uintptr_t>(slab->currentFree);
		// some wizard magic
		// (is just masking)
		uintptr_t alignedAddr = (currentAddr + alignment - 1) & ~(alignment - 1);
		uintptr_t slabEnd = reinterpret_cast<uintptr_t>(slab->start) + slab->size;

		if (alignedAddr + bytes <= slabEnd) [[likely]] {

			slab->currentFree = reinterpret_cast<uint8_t*>(alignedAddr + bytes);
			void* ptr = reinterpret_cast<void*>(alignedAddr);
			slab->allocatedBlocks++;
			return ptr;
		}

		activeSlab++;
		if (slabs.size() <= activeSlab)
			fillSlabs(REFILLSIZE);


		slab = slabs[activeSlab];

		currentAddr = reinterpret_cast<uintptr_t>(slab->currentFree);
		alignedAddr = (currentAddr + alignment - 1) & ~(alignment - 1);

		slab->currentFree = reinterpret_cast<uint8_t*>(alignedAddr + bytes);
		void* ptr = reinterpret_cast<void*>(alignedAddr);
		slab->allocatedBlocks++;

		return ptr;
	}

	inline void dealloc(void* ptr) {
		MediumSlab* slab =
		    reinterpret_cast<MediumSlab*>(reinterpret_cast<uintptr_t>(ptr) & (~(SLABSIZE - 1)));

		slab->allocatedBlocks--;

		if (slab->allocatedBlocks != 0)
			return;

		_CHECK_(activeSlab == 0);
		slab->currentFree = slab->start;
		std::swap(slabs[activeSlab - 1], slabs[slab->indexInSlabs]);

		slabs[activeSlab - 1]->indexInSlabs = activeSlab - 1;
		slabs[slab->indexInSlabs]->indexInSlabs = slab->indexInSlabs;
		activeSlab--;
	}

	~MediumAllocator() {
		for (MediumSlab* i : slabs) {
			free(i->start - SLABHEADERSIZE);
		}
	}

  private:
	inline void fillSlabs(uint16_t slabNum) {
		slabs.reserve(slabNum);
		for (uint32_t i = 0; i < slabNum; ++i) {
			MediumSlab* mem = static_cast<MediumSlab*>(std::aligned_alloc(SLABSIZE, SLABSIZE));
			_CHECK_(!mem);

			mem->start = reinterpret_cast<uint8_t*>(mem) + SLABHEADERSIZE;
			mem->currentFree = reinterpret_cast<uint8_t*>(mem) + SLABHEADERSIZE;
			mem->size = SLABSIZE - SLABHEADERSIZE;
			mem->allocatedBlocks = 0;
			mem->indexInSlabs = slabs.size();

			slabs.push_back(mem);
		}
	}
};

//& ================================================================================
//~    <- 4KB (end is inclusive)
class SmallAllocator {
  private:
	struct FreeBlock {
		FreeBlock* next;
	};

	struct SmallSlab {
		uint8_t* start;
		uint16_t blockSizeType;
		SmallSlab* next = nullptr;
		FreeBlock* freeList;
		uint32_t allocatedBlocks = 0;
	};

	//~ 16KB
	const inline static constexpr uint16_t REFILLSIZE = 2;
	const inline static constexpr uint16_t SLABHEADERSIZE = sizeof(SmallSlab);
	const inline static constexpr uintptr_t SLABSIZE = 16 * 1024; //! must be a power of two
	const inline static constexpr uint16_t POOLTYPENUMBER = 10;
	const inline static constexpr uint32_t POOLSIZE[POOLTYPENUMBER] = // the possible bite pools
	    {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
	const inline static constexpr uint16_t POOLWEIGHT[POOLTYPENUMBER] = // weights for distributing
	    {10, 13, 15, 15, 13, 10, 8, 8, 4, 4};                           // the memory when refilling
	                                                                    //* adds up to 100


	SmallSlab* globalBin_[POOLTYPENUMBER] = {nullptr};
	MediumAllocator& midAlloc_;

	//~ 1MB memory
	struct ThreadLocalCache {
		const static constexpr uint16_t SLABREFILL = 2;
		SmallSlab* bin[POOLTYPENUMBER] = {nullptr};
		SmallAllocator* allocator;

		void setAllocater(SmallAllocator& sAllocator) {
			allocator = &sAllocator;
		}

		~ThreadLocalCache() {
			if (!allocator)
				return;
			for (int i = 0; i < POOLTYPENUMBER; i++) {
				SmallSlab* slab = bin[i];
				while (slab) {
					SmallSlab* next = slab->next;
					if (slab->allocatedBlocks == 0) {
						slab->next = allocator->globalBin_[i];
						allocator->globalBin_[i] = slab;
					}
					slab = next;
				}
			}
		}
	};

	thread_local static ThreadLocalCache tlc_;


  public:
	SmallAllocator(const uint32_t startPoolSize, MediumAllocator& midAlloc) : midAlloc_(midAlloc) {
		if (startPoolSize == 0)
			return;

		fillPool(startPoolSize);
		fillTLC();
	}

	inline void* alloc(const size_t bytes) {

		// determening the sizeType
		uint16_t sizeType = POOLTYPENUMBER;
		for (uint16_t i = 0; i < POOLTYPENUMBER; ++i) {
			if (bytes <= POOLSIZE[i]) {
				sizeType = i;
				break;
			}
		}

		_CHECK_(sizeType == POOLTYPENUMBER);


		SmallSlab* slab = tlc_.bin[sizeType];
		if (slab) [[likely]] {
			_CHECK_(!slab->freeList);

			void* block = slab->freeList;
			slab->freeList = static_cast<FreeBlock*>(block)->next;
			slab->allocatedBlocks++;
			if (!slab->freeList) {
				tlc_.bin[sizeType] = slab->next;
			}

			return block;
		}


		fillTLC(sizeType);

		slab = tlc_.bin[sizeType];

		if (slab) [[likely]] {
			_CHECK_(!slab->freeList);

			void* block = slab->freeList;
			slab->freeList = static_cast<FreeBlock*>(block)->next;
			slab->allocatedBlocks++;
			if (!slab->freeList) {
				tlc_.bin[sizeType] = slab->next;
			}

			return block;
		}

		return midAlloc_.alloc(sizeType, std::min<uint32_t>(sizeType, 64));
	}

	inline void dealloc(void* ptr) {
		_CHECK(!ptr, "double free");
		SmallSlab* slab =
		    reinterpret_cast<SmallSlab*>(reinterpret_cast<uintptr_t>(ptr) & (~(SLABSIZE - 1)));

		FreeBlock* fb = reinterpret_cast<FreeBlock*>(ptr);
		fb->next = slab->freeList;
		slab->freeList = fb;

		slab->allocatedBlocks--;

		if (slab->allocatedBlocks != 0)
			return;


		slab->next = globalBin_[slab->blockSizeType];
		globalBin_[slab->blockSizeType] = slab;
	}


	~SmallAllocator() {
		for (int i = 0; i < POOLTYPENUMBER; i++) {
			while (globalBin_[i]) {
				SmallSlab* slab = globalBin_[i];
				globalBin_[i] = slab->next;
				midAlloc_.dealloc(slab);
			}
		}
	}

  private:
	friend class salloc;

	inline void fillTLC() {
		for (uint32_t i = 0; i < POOLTYPENUMBER; i++)
			fillTLC(i);
	}

	inline void fillTLC(const uint16_t sizeType) {
		//* can be much faster

		for (uint16_t i = 0; i < tlc_.SLABREFILL; i++) {
			if (!globalBin_[sizeType])
				fillPool(SLABSIZE, sizeType);
			_CHECK_(!globalBin_[sizeType] || !globalBin_[sizeType]->start);

			SmallSlab* slab = globalBin_[sizeType];

			globalBin_[sizeType] = slab->next;
			slab->next = tlc_.bin[sizeType];
			tlc_.bin[sizeType] = slab;
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
			const uint32_t blockSize = std::max<uint32_t>(POOLSIZE[sizeType], sizeof(FreeBlock));
			SmallSlab* slab = static_cast<SmallSlab*>(midAlloc_.alloc(SLABSIZE, SLABSIZE));

			_CHECK(!slab, "out of memory");

			slab->start = reinterpret_cast<uint8_t*>(slab) + SLABHEADERSIZE;
			slab->next = globalBin_[sizeType];
			globalBin_[sizeType] = slab;

			uint8_t* ptr = slab->start;
			uint8_t* end = ptr + SLABSIZE - SLABHEADERSIZE;

			while (ptr + blockSize <= end) {
				reinterpret_cast<FreeBlock*>(ptr)->next = slab->freeList;
				slab->freeList = reinterpret_cast<FreeBlock*>(ptr);

				ptr = ptr + blockSize;
			}
		}
	}
};
thread_local SmallAllocator::ThreadLocalCache SmallAllocator::tlc_;

//& ================================================================================
// the standard CPU allocater
// do not does the "dirty" work (allocating)
// splits the up the work
class salloc : Allocator {
  private:
	// percentiges of the allocators
	//! has to add up to 100%
	const inline static constexpr uint16_t INITRATIO[2] = {50, 50};

	LargeAllocator la_;  //~ 1MB >
	MediumAllocator ma_; //~ 4KB <  < 1Mb (end is inclusive)
	SmallAllocator sa_;  //~        < 4KB (end is inclusive)


  public:
	// first the medium then the small
	salloc(const uint32_t bitesToPool)
	    : ma_(bitesToPool * INITRATIO[1] / 100), sa_(bitesToPool * INITRATIO[0] / 100, ma_), la_() {
	}

	inline Device device() const override {
		return Device::CPU;
	};

	inline void* allocate(const size_t bytes, const size_t alignment = 64) override {
		// give the small alocater to SmallAllocator::tlc_
		sa_.tlc_.setAllocater(sa_);
		if (bytes <= 4 * 1024) {                //~ 0b
			return sa_.alloc(bytes);            //~
		} else if (bytes <= 1024 * 1024) {      //~ 4KB
			return ma_.alloc(bytes, alignment); //~
		} else {                                //~ 1Mb
			return la_.alloc(bytes, alignment); //~
		}
	};

	inline void deallocate(void*& ptr, const size_t bytes) override {
		if (bytes <= 4 * 1024) {            //~ 0b
			return sa_.dealloc(ptr);        //~
		} else if (bytes <= 1024 * 1024) {  //~ 4KB
			return ma_.dealloc(ptr);        //~
		} else {                            //~ 1Mb
			return la_.dealloc(ptr, bytes); //~
		}
		ptr = nullptr;
		return;
	};

	~salloc() = default;
};

class Buffer {
  private:
	void* data_;
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