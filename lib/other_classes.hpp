#pragma once

// TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

namespace TZ {

template <typename Derived, typename T>
class TensorWrapper {
  protected:
	Tensor<T> t_;

  public:
};

template <typename T>
class Matrix : public TensorWrapper<Matrix<T>, T> {
  public:
};

template <typename T>
class Vector : public TensorWrapper<Vector<T>, T> {
  public:
};

} // namespace TZ