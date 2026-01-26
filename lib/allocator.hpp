#pragma once

namespace TZ {
namespace mem {

enum Device { CPU, CUDA };

// an abstract class
class Allocator {
  public:
	virtual Device device() const = 0;

	virtual void* allocate(const size_t bytes, const size_t alignment) = 0;

	virtual void deallocate(void*& ptr, const size_t bytes) = 0;

	virtual ~Allocator() = default;

	Allocator() = default;


	Allocator(const Allocator&) = delete;

	Allocator& operator=(const Allocator&) = delete;

	Allocator(Allocator&&) = delete;

	Allocator& operator=(Allocator&&) = delete;
};

//& ================================================================================
//~ 1MB ->
class LargeAllocator {
  private:
	struct LargeBlock {
		size_t size = 0;
		LargeBlock* next = nullptr;
	};

	std::mutex mtx_;
	LargeBlock* blocks_ = nullptr;

	friend class salloc;

	LargeAllocator() = default;

  public:
	void* alloc(const size_t bytes, const size_t alignment) {
		std::lock_guard<std::mutex> lock(mtx_);
		LargeBlock* ptr = blocks_;
		LargeBlock* last = nullptr;

		while (ptr) {
			uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
			uintptr_t aligned = (raw + alignment - 1) & ~(alignment - 1);

			if (aligned + bytes <= raw + ptr->size) {
				if (last)
					last->next = ptr->next;
				else
					blocks_ = ptr->next;

				return reinterpret_cast<void*>(aligned);
			}
			last = ptr;
			ptr = ptr->next;
		}

		uint32_t size = ((bytes + alignment - 1) / alignment) * alignment;

		return std::aligned_alloc(alignment, size);
	}

	void dealloc(void* ptr, const size_t bytes) {
		std::lock_guard<std::mutex> lock(mtx_);
		LargeBlock* currBlock = reinterpret_cast<LargeBlock*>(ptr);
		currBlock->size = bytes;
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
};

//& ================================================================================
//~ 4KB - 1Mb (end is inclusive)
class MediumAllocator {
  private:
	struct MediumSlab {
		uint8_t* freeMem = nullptr;
		uint32_t allocatedBlocks = -1;
		size_t indexInBin = -1;
	};

	const static uintptr_t SLABSIZE = 4 * 1024 * 1024; //! must be a power of two
	const static uint16_t REFILLSIZE = 2;
	const static uint16_t SLABHEADERSIZE = 64;


	std::vector<MediumSlab*> bin_;
	std::atomic<uint16_t> activeSlab_ = 0; // index pointing to the next active slab

	MediumAllocator(const uint32_t startPoolSize) {
		if (startPoolSize == 0)
			return;

		uint16_t slabNum = (startPoolSize + SLABSIZE - 1) / SLABSIZE;

		fillSlabs(slabNum);
	}


  public:
	void* alloc(const size_t bytes, const size_t alignment) {
		while (true) {
			if (bin_.size() <= activeSlab_)
				fillSlabs(REFILLSIZE);


			MediumSlab* slab = bin_[activeSlab_];

			uintptr_t currentAddr = reinterpret_cast<uintptr_t>(slab->freeMem);
			uintptr_t alignedAddr = (currentAddr + alignment - 1) & ~(alignment - 1);
			uintptr_t slabEnd = reinterpret_cast<uintptr_t>(slab) + SLABSIZE;

			if (alignedAddr + bytes <= slabEnd) [[likely]] {

				slab->freeMem = reinterpret_cast<uint8_t*>(alignedAddr + bytes);
				void* ptr = reinterpret_cast<void*>(alignedAddr);
				slab->allocatedBlocks++;
				return ptr;
			}
			activeSlab_++;
		}

		return nullptr;
	}

	void dealloc(void* ptr) {
		MediumSlab* slab =
		    reinterpret_cast<MediumSlab*>(reinterpret_cast<uintptr_t>(ptr) & (~(SLABSIZE - 1)));

		slab->allocatedBlocks--;

		if (slab->allocatedBlocks != 0)
			return;

		if (activeSlab_ == 0)
			return;

		slab->freeMem = reinterpret_cast<uint8_t*>(slab) + SLABHEADERSIZE;
		std::swap(bin_[activeSlab_ - 1], bin_[slab->indexInBin]);

		bin_[activeSlab_ - 1]->indexInBin = activeSlab_ - 1;
		bin_[slab->indexInBin]->indexInBin = slab->indexInBin;
		activeSlab_--;
	}

	~MediumAllocator() {
		for (auto* i : bin_) {
			if (i->allocatedBlocks == 0)
				free(i);
		}
	}

  private:
	friend class salloc;

	void fillSlabs(uint16_t slabNum) {
		bin_.reserve(slabNum);
		for (uint32_t i = 0; i < slabNum; ++i) {
			MediumSlab* mem = static_cast<MediumSlab*>(std::aligned_alloc(SLABSIZE, SLABSIZE));
			_CHECK_(!mem);

			mem->freeMem = reinterpret_cast<uint8_t*>(mem) + SLABHEADERSIZE;
			mem->allocatedBlocks = 0;
			mem->indexInBin = bin_.size();

			bin_.push_back(mem);
		}
	}
};

//& ================================================================================
//~    <- 4KB (end is inclusive)
class SmallAllocator {
  private:
	struct FreeBlock {
		FreeBlock* next = nullptr;
	};

	struct SmallSlab {
		uint16_t blockSizeType = -1;
		SmallSlab* nextSlab = nullptr;
		FreeBlock* nextFreeBlock = nullptr;
		uint32_t allocatedBlocks = 0;
		bool notAvailable = false;
	};

	//~ 16KB
	const static uint16_t REFILLSIZE = 2;
	const static uint16_t SLABHEADERSIZE = 64;   // sizeof(SmallSlab)
	const static uintptr_t SLABSIZE = 32 * 1024; //! must be a power of two
	const static uint16_t POOLTYPENUMBER = 10;
	const static constexpr uint32_t POOLSIZE[POOLTYPENUMBER] = // the possible bite pools
	    {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
	const static constexpr uint16_t POOLWEIGHT[POOLTYPENUMBER] = // weights for distributing
	    {6, 8, 10, 10, 10, 10, 10, 10, 12, 14};                  // the memory when refilling
	                                                             //* adds up to 100


	SmallSlab* bin_[POOLTYPENUMBER] = {nullptr};
	MediumAllocator& midAlloc_;

	SmallAllocator(const uint32_t startPoolSize, MediumAllocator& midAlloc) : midAlloc_(midAlloc) {
		if (startPoolSize == 0)
			return;

		fillPool(startPoolSize);
	}

  public:
	void* alloc(const size_t bytes) {

		if (bytes == 0)
			return nullptr;

		// determening the sizeType
		uint16_t sizeType = POOLTYPENUMBER;
		for (uint16_t i = 0; i < POOLTYPENUMBER; ++i) {
			if (bytes <= POOLSIZE[i]) {
				sizeType = i;
				break;
			}
		}

		_CHECK_(sizeType == POOLTYPENUMBER);

		while (true) {
			SmallSlab* slab = bin_[sizeType];
			if (slab) [[likely]] {
				_CHECK_(!slab->nextFreeBlock);

				void* block = slab->nextFreeBlock;
				slab->nextFreeBlock = slab->nextFreeBlock->next;
				slab->allocatedBlocks++;
				if (!slab->nextFreeBlock) {
					slab->notAvailable = true;
					bin_[sizeType] = slab->nextSlab;
				}

				return block;
			}
			fillPool(REFILLSIZE * SLABSIZE, sizeType);
		}
		return nullptr;
	}

	void dealloc(void* ptr) {
		_CHECK(!ptr, "double free");
		SmallSlab* slab =
		    reinterpret_cast<SmallSlab*>(reinterpret_cast<uintptr_t>(ptr) & (~(SLABSIZE - 1)));

		FreeBlock* block = reinterpret_cast<FreeBlock*>(ptr);
		block->next = slab->nextFreeBlock;
		slab->nextFreeBlock = block;

		slab->allocatedBlocks--;

		if (slab->allocatedBlocks != 0 || !slab->notAvailable)
			return;

		slab->notAvailable = false;
		slab->nextSlab = bin_[slab->blockSizeType];
		bin_[slab->blockSizeType] = slab;
	}

	~SmallAllocator() {
		for (int t = 0; t < POOLTYPENUMBER; t++) {
			SmallSlab* slab = bin_[t];
			while (slab) {
				SmallSlab* next = slab->nextSlab;
				if (slab->allocatedBlocks == 0)
					midAlloc_.dealloc(slab);
				slab = next;
			}
		}
	}

  private:
	friend class salloc;

	void fillPool(const uint32_t bites) {
		for (uint32_t i = 0; i < POOLTYPENUMBER; i++)
			fillPool((bites * POOLWEIGHT[i]) / 100, i);
	}

	void fillPool(uint32_t bites, const uint16_t sizeType) {
		//* can be much faster
		// if the allocated space is to small for a slab round it up to 1
		uint32_t numSlabs = (bites + SLABSIZE - 1) / SLABSIZE;


		for (uint32_t i = 0; i < numSlabs; i++) {
			const uint32_t blockSize = std::max<uint32_t>(POOLSIZE[sizeType], sizeof(FreeBlock));
			SmallSlab* slab = static_cast<SmallSlab*>(midAlloc_.alloc(SLABSIZE, SLABSIZE));

			_CHECK(!slab, "out of memory");

			slab->nextSlab = bin_[sizeType];
			slab->blockSizeType = sizeType;
			bin_[sizeType] = slab;

			uint8_t* ptr = reinterpret_cast<uint8_t*>(slab) + SLABHEADERSIZE;
			uint8_t* end = ptr + SLABSIZE - SLABHEADERSIZE;

			while (ptr + blockSize <= end) {
				reinterpret_cast<FreeBlock*>(ptr)->next = slab->nextFreeBlock;
				slab->nextFreeBlock = reinterpret_cast<FreeBlock*>(ptr);

				ptr = ptr + blockSize;
			}
		}
	}
};

//& ================================================================================
// the standard CPU allocater
//! singelton
class salloc : Allocator {
  private:
	// percentiges of the allocators
	//! has to add up to 100%
	const static constexpr uint16_t INITRATIO[2] = {50, 50};

	LargeAllocator la_; //~ 1MB >

	MediumAllocator& ma_() {
		static thread_local MediumAllocator ma_(START_MEM_SIZE * INITRATIO[1] / 100);
		return ma_;
	}

	SmallAllocator& sa_() {
		static thread_local SmallAllocator sa_(START_MEM_SIZE * INITRATIO[0] / 100, ma_());
		return sa_;
	}

	salloc() = default;


  public:
	static salloc& instance() {
		static salloc* alloc = new salloc;
		return *alloc;
	}

	Device device() const override {
		return Device::CPU;
	};

	void* allocate(const size_t bytes, const size_t alignment = 64) override {
		if (bytes <= 4 * 1024) {                  //~ 0b
			return sa_().alloc(bytes);            //~
		} else if (bytes <= 1024 * 1024) {        //~ 4KB
			return ma_().alloc(bytes, alignment); //~
		} else {                                  //~ 1Mb
			return la_.alloc(bytes, alignment);   //~
		}
	};

	void deallocate(void*& ptr, const size_t bytes) override {
		if (bytes <= 4 * 1024) {            //~ 0b
			return sa_().dealloc(ptr);      //~
		} else if (bytes <= 1024 * 1024) {  //~ 4KB
			return ma_().dealloc(ptr);      //~
		} else {                            //~ 1Mb
			return la_.dealloc(ptr, bytes); //~
		}
		ptr = nullptr;
		return;
	};

	~salloc() = default;


	salloc(const salloc&) = delete;

	salloc& operator=(const salloc&) = delete;

	salloc(salloc&&) = delete;

	salloc& operator=(salloc&&) = delete;
};

// standard memory buffer
class Buffer {
  private:
	void* data_;
	const size_t size_;
	const uint8_t alignment_;
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