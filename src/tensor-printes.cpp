#include <iostream>
#include "special_functions.hpp"
#include "Tenzor.hpp"

namespace TZ{

    template<typename T>
    void Tensor<T>::printShape(const std::string &name) const{
        if(shape.size() == 0){
            std::cout << name << " is scalar" << std::endl;
            return;
        }

        std::cout << name << "Shape: \n{ " << shape[0];
        for(int i = 1; i < shape.size(); i++)
            std::cout << ", " << shape[i];
        
        std::cout << "}" << std::endl;
        
    }
};