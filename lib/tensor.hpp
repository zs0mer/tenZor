#pragma once

namespace TZ {

template <typename T> class Tensor {
  private:
	std::vector<uint64_t> shape_;
	std::vector<uint64_t> strides_;
	uint64_t offset_ = 0;
	mem::Buffer data_;

  public:
	Tensor(const std::vector<uint64_t>& shape, uint8_t alignment = 64, mem::Allocator allocator);
};

} // namespace TZ