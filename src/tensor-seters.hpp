#pragma once
#include "tenzor_utils.hpp"
#include "tensor.hpp"

namespace TZ{
    // -------------------------------------------------------
    // constructors

    // the standard constructor
    template<typename T>
    Tensor<T>::Tensor(const int &newDimenson, const std::vector<int> &newShape, const std::vector<T> &newData){set(newDimenson, newShape, newData);}

    // using nested std::vectors
    template<typename T>
    template<typename NestedVector>
    Tensor<T>::Tensor(const NestedVector &newData){set(newData);}

    // -------------------------------------------------------
    // setters

    // standard set implement
    template<typename T>
    void Tensor<T>::set(const int &newDimenson, const std::vector<int> &newShape, const std::vector<T> &newData){
        // check if input is correct
        CHECK(newDimenson != newShape.size(), "missmach beetwen new dimension and new shape-array size");

        long long elementNumber = 1;
        for(int& i : newShape)
            elementNumber *= i;
        
        CHECK(elementNumber != newData.size(), "missmach beetwen new data size and expected size");

        // sets the input
        dimension = newDimenson;
        shape = newShape;
        data = newData;
    }


    // -----
    // unrolling nested std::vectors
    template <typename T>
    template<typename NestedVector>
    void Tensor<T>::set(const NestedVector &newData){
        shape = {};
        data = {};
        discover(newData, shape);
        unroll(newData, data, 0, shape);
        dimension = shape.size();
    }

    // recursive function
    template <typename V>
    static void discover(const std::vector<V> &v, std::vector<int> &shape) {
        TZ_check_impl(v[0].size() == 0, "nested std::vector size is 0", __FILE__, __LINE__, __func__);
        shape.push_back(v.size());
        discover(v[0], shape);
    }
    
    // base case
    template <typename T>
    static void discover(const T &arr, std::vector<int> &shape) {}
    
    
    // recursive function
    template <typename V,typename T>
    static void unroll(const std::vector<V> &v, std::vector<T> &out, int depth, const std::vector<int> &shape) {
        TZ_check_impl(shape[depth] != v.size(), "missmaching array size when seting up the tensor from nested std::vectors", __FILE__, __LINE__, __func__);
        for (const auto& i : v) unroll(i, out, depth+1, shape);
    }
    
    // base case
    template <typename T>
    static void unroll(const T &v, std::vector<T> &out, int depth, const std::vector<int> &shape){out.push_back(v);}
    // -----
    

};