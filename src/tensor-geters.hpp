#pragma once
#include "tensor.hpp"
// ^ this is NOT esential, I just use this so VScode does not get confused

namespace TZ{

    // returns the dimenson
    template<typename T>
    int Tensor<T>::getDimenson() const{
        return dimension;
    };

    // returns the copy of the shape array
    template<typename T>
    std::vector<int> Tensor<T>::getShape() const{
        return shape;
    };

    //* standard get element functions with std::vector<int>
    template<typename T>
    T& Tensor<T>::get(const std::vector<int> &index){
        CHECK(index.size() != dimension, "missmaching get array dimenson");
        long long idx = 0;

        for(int i = 0; i < dimension; i++){
            idx *= shape[i];
            idx += index[i];
        }

        CHECK(idx >= data.size(), "to large index");

        return data[idx];
    }

    template<typename T>
    const T& Tensor<T>::get(const std::vector<int> &index) const{
        return get(index);
    }

    template<typename T>
    T& Tensor<T>::operator[](const std::vector<int>& index){
        return get(index);
    }
    
    template<typename T>
    const T& Tensor<T>::operator[](const std::vector<int>& index)const{
        return get(index);
    }

    //* standard get element functions with std::initializer_list
    template<typename T>
    T& Tensor<T>::get(const std::initializer_list<int> &index){
        return get(std::vector<int>(index));
    }

    template<typename T>
    const T& Tensor<T>::get(const std::initializer_list<int> &index) const{
        return get(std::vector<int>(index));
    }

    template<typename T>
    T& Tensor<T>::operator[](const std::initializer_list<int>& index){
        return get(std::vector<int>(index));
    }
    
    template<typename T>
    const T& Tensor<T>::operator[](const std::initializer_list<int>& index)const{
        return get(std::vector<int>(index));
    }
    

};