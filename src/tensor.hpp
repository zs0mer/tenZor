#pragma once
#include <vector>
#include <string>

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

        // a way to get elements faster in the tensor
        // the first one is not important
        std::vector<long long> lookup;

        void calcLookup();

        public:

        // cals the standard set method
        explicit Tensor(const std::vector<int> &newShape, const std::vector<T> &newData);

        // just cals the set method that unroll a nested std::vector
        template <typename NestedVector>
        Tensor(const NestedVector &newData);

        // just make a new vector with shape newShape filled with vall
        Tensor(const std::vector<int> &newShape, const int &vall = 0);
        


        // standard set with flat std::vector
        void set(const std::vector<int> &newShape, const std::vector<T> &newData);

        // set with multiple std::vectors nested
        // for example: {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
        template <typename NestedVector>
        void set(const NestedVector &newData);

        void set(const std::vector<int> &newShape, const int &vall = 0);

        // returns the dimension of the tensor
        int getDimenson()const;

        // returns the shape of the tensor
        std::vector<int> getShape()const;

        // returns a reference to the element with that index
        // example: {0, 3, 4}
        // 3D tensor, same as tensor[0][3][4]
        // O(number of dimensons)

        //* using std::vector
        T& get(const std::vector<int> &index);

        const T& get(const std::vector<int> &index)const;

        T& operator[](const std::vector<int>& index);
    
        const T& operator[](const std::vector<int>& index)const;
        

        //* using std::initializer_list

        T& get(const std::initializer_list<int> &index);

        const T& get(const std::initializer_list<int> &index)const;

        T& operator[](const std::initializer_list<int>& index);
    
        const T& operator[](const std::initializer_list<int>& index)const;

        // --

        // prints the shape of the tensor like this:
        // {name}
        // Shape: {a, b, c, ...}
        // if the tensor scalar then prints:
        // {name} is scalar
        void printShape(const std::string &name = "tensor")const;


        private:

        void discover(const T &arr, std::vector<int> &shape);

        template <typename V>
        void discover(const std::vector<V> &v, std::vector<int> &shape);

        void unroll(const T &v, std::vector<T> &out, int depth, const std::vector<int> &shape);

        template <typename V>
        void unroll(const std::vector<V> &v, std::vector<T> &out, int depth, const std::vector<int> &shape);
    };
    
};