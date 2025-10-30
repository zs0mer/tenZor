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


    // this is to flaten a N-D nested std::vector
    // we will use a recursive aproach

    template <typename E,typename X>
    void unroll(const std::vector<E>& v,std::vector<X>& out){
        std::cout << "unroll vector\n";
        out.insert(out.end(), v.begin(), v.end());
    }
    
    
    template <typename V,typename X>
    void unroll(const std::vector<std::vector<V>>& v,std::vector<X>& out) {
        std::cout << "unroll vector of vectors\n";
        for (const auto& e : v) unroll(e,out);
    }
    
    
    template <typename T>
    template<typename NestedVector>
    void tensor<T>::set(const NestedVector &newData){

    }


    template<typename T>
    tensor<T>::tensor(const int &newDimenson, const std::vector<int> &newShape, const std::vector<T> &newData) : set(newDimenson, newShape, newData){}
};