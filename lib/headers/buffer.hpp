#pragma once

#include <cstring>
#include <cstdint>
#include <atomic>

#include "allocator.hpp"

namespace TZ::mem {

// Buffer implementation
// holds the metadata for the buffer
class BufferIMPL {
  private:
	Allocator* const allocator_;
	const uint64_t size_;
	const uint8_t alignment_;
	std::atomic<uint32_t> refCount_{1};
	void* data_;

  public:
	BufferIMPL() = delete;

	// standard constructor
	// you can only construct with this constructor
	BufferIMPL(const uint64_t size, const uint8_t alignment = DEFAULT_ALIGNMENT,
	           Allocator* allocator = &defaultAllocator())
	    : size_(size), alignment_(alignment), allocator_(allocator),
	      data_(allocator->allocate(size, alignment)) {}

	// this makes a buffer with the same size, and data
	BufferIMPL(const BufferIMPL& other)
	    : size_(other.size_), alignment_(other.alignment_), allocator_(other.allocator_),
	      data_(allocator_->allocate(size_, alignment_)) {
		memcpy(data_, other.data_, size_);
	}

	BufferIMPL& operator=(const BufferIMPL&) = delete;

	BufferIMPL(BufferIMPL&&) = delete;

	BufferIMPL& operator=(BufferIMPL&&) = delete;

	// # lifetime----

	// increment the reference count
	void retain() noexcept {
		refCount_.fetch_add(1, std::memory_order_relaxed);
	}

	// decrement the reference count
	// if 0 deallocate
	bool release() noexcept {
		if (refCount_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			if (data_)
				allocator_->deallocate(data_, size_);
			return true;
		}
		return false;
	}

	// # data--------

	// returns the pointer to the buffer
	void* data() noexcept {
		return data_;
	};

	// returns the pointer to the buffer
	const void* data() const noexcept {
		return data_;
	};

	// # metadata----

	// returns size of the buffer
	uint64_t size() const noexcept {
		return size_;
	};

	// returns the device that this data is allocated on
	Device device() const noexcept {
		return allocator_->device();
	};

	// returns the allocator that this buffer is using
	Allocator* allocator() const noexcept {
		return allocator_;
	}
};

// # ================================================================================

// standard memory buffer
// holds the BufferIMPL
class Buffer {
	BufferIMPL* ptr_;

  public:
	// this buffer will become the new owner of the BufferIMPL
	Buffer(BufferIMPL* buffer)
	    : ptr_(buffer ? buffer
	                  : static_cast<BufferIMPL*>(defaultAllocator(CPU).allocate(
	                        sizeof(BufferIMPL), DEFAULT_ALIGNMENT))) {
		new (ptr_) BufferIMPL(0, DEFAULT_ALIGNMENT, &defaultAllocator());
	}

	// standard constructor
	Buffer(const uint64_t size = 0, Allocator* allocator = &defaultAllocator())
	    : ptr_(static_cast<BufferIMPL*>( // allocate with the default CPU allocater
	          defaultAllocator(CPU).allocate(sizeof(BufferIMPL), DEFAULT_ALIGNMENT))) {
		new (ptr_) BufferIMPL(size, DEFAULT_ALIGNMENT, allocator);
	}

	// this buffer will contain the same BufferIMPL
	Buffer(const Buffer& other) : ptr_(other.ptr_) {
		if (ptr_)
			ptr_->retain();
	}

	// this buffer will contain the same BufferIMPL
	Buffer& operator=(const Buffer& other) {
		if (this == &other)
			return *this;

		clear();

		ptr_ = other.ptr_;
		if (ptr_)
			ptr_->retain();

		return *this;
	}

	// this buffer will own the BufferIMPL inside the other Buffer
	Buffer(Buffer&& other) : ptr_(other.ptr_) {
		other.ptr_ = nullptr;
	}

	// this buffer will own the BufferIMPL inside the other Buffer
	Buffer& operator=(Buffer&& other) {
		if (this == &other)
			return *this;

		clear();

		ptr_ = other.ptr_;
		other.ptr_ = nullptr;

		return *this;
	}

	~Buffer() {
		clear();
	}

	BufferIMPL* operator->() {
		return ptr_;
	}

	const BufferIMPL* operator->() const {
		return ptr_;
	}

	// makes a new buffer with the same data
	Buffer clone() const {
		BufferIMPL* p = static_cast<BufferIMPL*>(
		    ptr_->allocator()->allocate(sizeof(BufferIMPL), DEFAULT_ALIGNMENT));
		new (p) BufferIMPL(*ptr_);
		return Buffer(p);
	}

  private:
	void clear() {
		if (ptr_)
			if (ptr_->release())
				defaultAllocator(CPU).deallocate(static_cast<void*>(ptr_), sizeof(BufferIMPL));
	}
};


} // namespace TZ::mem
