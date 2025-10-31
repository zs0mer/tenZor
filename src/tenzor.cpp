#include <string>
#include <iostream>
#include <ctime>
#include "Tenzor.hpp"

// a static check function to make error handleing easier
inline static void check(const bool expresson, const std::string error = "unexpected"){
    if(expresson){
        std::time_t now = std::time(nullptr);
        throw std::runtime_error("Error: " + error + ". \n time: " + std::ctime(&now));
    }
}


namespace TZ{

    // standard set implement
    template<typename T>
    void tensor<T>::set(const int &newDimenson, const std::vector<int> &newShape, const std::vector<T> &newData){
        // check if input is correct
        check(newDimenson != newShape.size(), "missmach beetwen new dimenson and new shape-array size");

        long long elementNumber = 1;
        for(int& i : newShape)
            elementNumber *= i;
        
        check(elementNumber != newData.size(), "missmach beetwen new data size and expected size");

        // sets the input
        dimenson = newDimenson;
        shape = newShape;
        data = newData;
    }


    // --------------------------------------------------------
    // this is to flaten a N-D nested std::vector
    // we will use a recursive aproach

    template <typename T>
    template<typename NestedVector>
    void tensor<T>::set(const NestedVector &newData){
        shape = {};
        data = {};
        discover(newData, shape);
        unroll(newData, data);
        dimenson = shape;
    }

    // recursive function
    template <typename V>
    void discover(const std::vector<std::vector<V>> &v, std::vector<int> &shape) {
        shape.push_back(v.size());
        discover(v[0], shape);
    }

    // base case
    template <typename T>
    void discover(const std::vector<T> &arr, std::vector<int> &shape) {
        shape.push_back(arr.size())
    }

    
    // recursive function
    template <typename V,typename T>
    void unroll(const std::vector<std::vector<V>> &v, std::vector<T> &out, int depth, const std::vector<int> &shape) {
        check(shape[depth] != v.size(), "missmaching array size when seting up the tensor from nested std::vectors");
        for (const auto& i : v) unroll(i, out, depth+1, shape);
    }

    // base case
    template <typename T>
    void unroll(const std::vector<T> &arr, std::vector<T> &out, int depth, const std::vector<int> &shape){
        check(shape[depth] != arr.size(), missmaching array size when seting up the tensor from nested std::vectors);
        out.insert(out.end(), arr.begin(), arr.end());
    }
    
    // --------------------------------------------------------



    template<typename T>
    tensor<T>::tensor(const int &newDimenson, const std::vector<int> &newShape, const std::vector<T> &newData) : set(newDimenson, newShape, newData){}
};