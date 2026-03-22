#pragma once

namespace TZ {

template <class T>
template <typename Func>
void Tensor<T>::apply(Tensor<T>& a, Func func) {}

template <class T>
template <typename Func>
void Tensor<T>::apply(const Tensor<T>& a, Tensor<T>& b, Func func) {}

template <class T>
template <typename Func>
void Tensor<T>::apply(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& c, Func func) {}

}; // namespace TZ