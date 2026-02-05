#pragma once

namespace TZ {

template <class T> size_t Tensor<T>::dim() const {}

template <class T> const std::vector<uint64_t>& Tensor<T>::shape() const {}

template <class T> T* Tensor<T>::data() {}

template <class T> const T* Tensor<T>::data() const {}

template <class T> bool Tensor<T>::empty() const {}

template <class T> bool Tensor<T>::is_contiguous() const {}

template <class T> bool Tensor<T>::isFullBuffer() const {}

template <class T> bool Tensor<T>::isScalar() const {}


} // namespace TZ
