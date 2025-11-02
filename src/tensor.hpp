#pragma once
#include <vector>
#include <string>
#include "tenzor_utils.hpp"

// --------------------------------------------------
// helper functions for the setter



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
        explicit Tensor(const int &newDimenson, const std::vector<int> &newShape, const std::vector<T> &newData);

        // just cals the set method that unroll a nested std::vector
        template <typename NestedVector>
        Tensor(const NestedVector &newData);
        


        // standard set with flat std::vector
        void set(const int &newDimenson, const std::vector<int> &newShape, const std::vector<T> &newData);

        // set with multiple std::vectors nested
        // for example: {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
        template <typename NestedVector>
        void set(const NestedVector &newData);

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