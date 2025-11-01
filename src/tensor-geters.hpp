#pragma once
#include "tenzor_utils.hpp"
#include "tensor.hpp"

namespace TZ{

    template<typename T>
    int Tensor<T>::getDimenson() const{
        return dimension;
    };

    template<typename T>
    std::vector<int> Tensor<T>::getShape() const{
        return shape;
    };
};