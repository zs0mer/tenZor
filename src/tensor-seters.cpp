#include "special_functions.hpp"
#include "Tenzor.hpp"

namespace TZ{
    // -------------------------------------------------------
    // constructors

    // the standard constructor
    template<typename T>
    Tensor<T>::Tensor(const int &newDimenson, const std::vector<int> &newShape, const std::vector<T> &newData){set(newDimenson, newShape, newData);}

    // implementation is in the header
    /*template<typename T>
    template<typename NestedVector>
    Tensor<T>::Tensor(const NestedVector &newData)*/

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


    // implementation is in the header
    /*template <typename T>
    template<typename NestedVector>
    void Tensor<T>::set(const NestedVector &newData)*/


    

};