#pragma once

namespace TZ {
namespace mem {

//~ Buffer implementation
class BufferIMPL {
  private:
	Allocator* const allocator_;
	const uint64_t size_;
	const uint8_t alignment_;
	std::atomic<uint32_t> refCount_{1};
	void* data_;

  public:
	BufferIMPL() = delete;

	BufferIMPL(const uint64_t size, const uint8_t alignment,
	           Allocator* allocator = &salloc::instance())
	    : size_(size), alignment_(alignment), allocator_(allocator),
	      data_(allocator->allocate(size, alignment)) {}

	BufferIMPL(const BufferIMPL& other)
	    : size_(other.size_), alignment_(other.alignment_), allocator_(other.allocator_),
	      data_(allocator_->allocate(size_, alignment_)) {
		memcpy(data_, other.data_, size_);
	}

	BufferIMPL& operator=(const BufferIMPL&) = delete;

	BufferIMPL(BufferIMPL&&) = delete;

	BufferIMPL& operator=(BufferIMPL&&) = delete;

	//& lifetime----

	void retain() noexcept {
		refCount_.fetch_add(1, std::memory_order_relaxed);
	}

	bool release() noexcept {
		if (refCount_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			allocator_->deallocate(data_, size_);
			return true;
		}
		return false;
	}

	//& data--------

	void* data() noexcept {
		return data_;
	};

	const void* data() const noexcept {
		return data_;
	};

	//& metadata----

	uint64_t size() const noexcept {
		return size_;
	};

	Device device() const noexcept {
		return allocator_->device();
	};

	Allocator* allocator() const noexcept {
		return allocator_;
	}
};

//& ================================================================================

//~ standard memory buffer
//~ holds the BufferIMPL
class Buffer {
	BufferIMPL* ptr_;

  public:
	Buffer(BufferIMPL* buffer) : ptr_(buffer) {}

	Buffer(const uint64_t size = 0, const uint8_t alignment = 64,
	       Allocator* allocator = &salloc::instance())
	    : ptr_(size == 0 ? nullptr : new BufferIMPL(size, alignment, allocator)) {}

	Buffer(const Buffer& other) : ptr_(other.ptr_) {
		if (ptr_)
			ptr_->retain();
	}

	Buffer& operator=(const Buffer& other) {
		if (this == &other)
			return *this;

		clear();

		ptr_ = other.ptr_;
		if (ptr_)
			ptr_->retain();

		return *this;
	}

	Buffer(Buffer&& other) : ptr_(other.ptr_) {
		other.ptr_ = nullptr;
	}

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


	Buffer clone() const {
		return Buffer(new BufferIMPL(*ptr_));
	}

  private:
	void clear() {
		if (ptr_)
			if (ptr_->release())
				delete ptr_;
	}
};

} // namespace mem
} // namespace TZ