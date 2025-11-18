#pragma once
#include "tensor.hpp"
// ^ this is NOT esential, I just use this so VScode does not get confused


namespace TZ{
    // -------------------------------------------------------
    // constructors

    // the standard constructor
    template<typename T>
    Tensor<T>::Tensor(const std::vector<int> &newShape, const std::vector<T> &newData){
        set(newShape, newData);
    }

    // using nested std::vectors
    template<typename T>
    template<typename NestedVector>
    Tensor<T>::Tensor(const NestedVector &newData){
        set(newData);
    }

    template<typename T>
    Tensor<T>::Tensor(const std::vector<int> &newShape, const int &vall){
        set(newShape, vall);
    }

    // -------------------------------------------------------
    // setters

    // standard set implement
    template<typename T>
    void Tensor<T>::set(const std::vector<int> &newShape, const std::vector<T> &newData){

        long long size = 1;
        for(const int& i : newShape)
            size *= i;
        
        CHECK(size != newData.size(), "missmach beetwen new data size and expected size");

        // sets the input
        dimension = newShape.size();
        shape = newShape;
        data = newData;
        calcLookup();
    }


    // -----
    //* set but with one std::vectors
    // unrolling nested std::vectors

    // base case
    template <typename T>
    void Tensor<T>::discover(const T &arr, std::vector<int> &shape) {}

    // recursive function
    template <typename T>
    template <typename V>
    void Tensor<T>::discover(const std::vector<V> &v, std::vector<int> &shape) {
        CHECK(v.size() == 0, "nested std::vector size is 0");
        shape.push_back(v.size());
        discover(v[0], shape);
    }
    
    
    // base case
    template <typename T>
    void Tensor<T>::unroll(const T &v, std::vector<T> &out, int depth, const std::vector<int> &shape){out.push_back(v);}

    // recursive function
    template <typename T>
    template <typename V>
    void Tensor<T>::unroll(const std::vector<V> &v, std::vector<T> &out, int depth, const std::vector<int> &shape) {
        CHECK(shape[depth] != v.size(), "missmaching array size when seting up the tensor from nested std::vectors");
        for (const auto& i : v) unroll(i, out, depth+1, shape);
    }

    template <typename T>
    template<typename NestedVector>
    void Tensor<T>::set(const NestedVector &newData){
        shape = {};
        data = {};
        discover(newData, shape);
        unroll(newData, data, 0, shape);
        dimension = shape.size();
        calcLookup();
    }

    // -----

    template<typename T>
    void Tensor<T>::set(const std::vector<int> &newShape, const int &vall){
        dimension = newShape.size();
        shape = newShape;
        
        long long size = 1;
        for(int &i : shape)
            size *= i;

        data.assign(size, vall);
        calcLookup();
    }


    // when changing the tensor we have to recalculate the lookup vals
    template<typename T>
    void Tensor<T>::calcLookup(){
        long long size = 0;
        for(int i = dimension-2; i >= 0; i--)
            size = (size + 1) * shape[i];

        size++;
        lookup.assign(size, 1);

        long long t = 1;
        long long k = 1;
        for(int i = 0; i < dimension-2; i++){
            for(int j = 0; j < shape[i]*k; j++)
                lookup[t+j] = j*shape[i+1] + (t + k*shape[i]);
            
            t += k*shape[i];
            k = k*shape[i];
        }

        // this is only for the last segment, to get the correct index
        for(int j = 0; j < shape[dimension-2]*k; j++)
                lookup[t+j] = j*shape[dimension-1];
    }

    

};