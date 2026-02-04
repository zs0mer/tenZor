#pragma once

namespace TZ {
template <class T>
Tensor<T>::Tensor(const std::vector<uint64_t>& shape, const uint8_t alignment,
                  mem::Allocator& allocator) {}
/*
template <class nestedVector, class T>
Tensor<T>::Tensor(const std::vector<nestedVector>& nestedVector, const uint8_t alignment,
                  mem::Allocator& allocator) {}

template <class nestedList, class T>
Tensor<T>::Tensor(const std::initializer_list<nestedList> nestedVector, const uint8_t alignment,
                  mem::Allocator& allocator) {}
*/
template <class T>
Tensor<T>::Tensor(const std::vector<uint64_t>& shape_, const mem::Buffer data_,
                  const std::vector<uint64_t>& strides_, const uint64_t offset_) {}

template <class T> Tensor<T>::Tensor(const Tensor& other) {}

template <class T> Tensor& Tensor<T>::operator=(const Tensor& other) {}

template <class T> Tensor<T>::Tensor(Tensor&& other) {}

template <class T> Tensor& Tensor<T>::operator=(Tensor&& other) {}
} // namespace TZ