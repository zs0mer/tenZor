#pragma once


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
    }


    // -----
    //* set but with one std::vectors
    // unrolling nested std::vectors

    // base case
    template <typename T>
    static void discover(const T &arr, std::vector<int> &shape) {}

    // recursive function
    template <typename V>
    static void discover(const std::vector<V> &v, std::vector<int> &shape) {
        CHECK(v.size() == 0, "nested std::vector size is 0");
        shape.push_back(v.size());
        discover(v[0], shape);
    }
    
    
    // base case
    template <typename T>
    static void unroll(const T &v, std::vector<T> &out, int depth, const std::vector<int> &shape){out.push_back(v);}

    // recursive function
    template <typename V,typename T>
    static void unroll(const std::vector<V> &v, std::vector<T> &out, int depth, const std::vector<int> &shape) {
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
    }


};