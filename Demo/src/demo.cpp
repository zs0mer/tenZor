#include "Tenzor.hpp"


int main(){
    TZ::Tensor<int32_t> t(std::vector<std::vector<int>>{{3,2},{2,3},{1,4}});
    std::cout << t[{0, 1}];
    return 0;
}