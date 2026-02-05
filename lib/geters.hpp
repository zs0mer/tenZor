#pragma once

namespace TZ {

template <class T> T& Tensor<T>::at(const std::vector<uint64_t>& idx) {}

template <class T> const T& Tensor<T>::at(const std::vector<uint64_t>& idx) const {}

template <class T> Tensor<T> Tensor<T>::operator[](const uint64_t idx) const {}

template <class T> Tensor<T> Tensor<T>::clone() const {}

template <class T> T Tensor<T>::scalarVal() const {}

} // namespace TZ