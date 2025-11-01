#pragma once
#include <vector>
#include <string>

inline static void _check(const bool expresson, const std::string& error, const char* file, int line, const char* func);

// --------------------------------------------------
// helper functions for the setter

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
 
// --------------------------------------------------

namespace TZ {

    /*
    
    ##############################
    ######## Tensor class ########
    ##############################
    
    */

    template<typename T>
    class Tensor
    {
    private:

        // the dimension of the tensor
        // if 0 the object functions as a scalar
        int dimension; 

        // the shape of the tensor
        std::vector<int> shape;

        // the stored data
        // using a singel dimension
        // size = shape[0] * shape[1] ... shape[n]
        // data[i][j]...[k] = data[i * (shape[1] * shape[2] ... shape[n]) + j * (shape[2] * shape[3] ... shape[n]) ... + k]   (I think)
        std::vector<T> data;

        public:

        // cals the standard set method
        Tensor(const int &newDimenson, const std::vector<int> &newShape, const std::vector<T> &newData);

        // just cals the set method that unroll a nested std::vector
        // sadly I have to put the definition in here as well
        template <typename NestedVector>
        Tensor(const NestedVector &newData){set(newData);}

        // standard set with flat std::vector
        void set(const int &newDimenson, const std::vector<int> &newShape, const std::vector<T> &newData);

        // set with multiple std::vectors nested
        // for example: {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
        // sadly I have to put the definition in here as well
        template <typename NestedVector>
        void set(const NestedVector &newData){
            shape = {};
            data = {};
            discover(newData, shape);
            unroll(newData, data, 0, shape);
            dimension = shape.size();
        }

        // returns the dimension of the tensor
        int getDimenson()const;

        // returns the shape of the tensor
        std::vector<int> getShape()const;

        // prints the shape of the tensor like this:
        // {name}
        // Shape: {a, b, c, ...}
        // if the tensor scalar then prints:
        // {name} is scalar
        void printShape(const std::string &name = "tensor")const;
    };
    
};