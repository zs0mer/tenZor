#include "Tenzor.hpp"


int main(){
    TZ::Tensor<int32_t> t(std::vector<std::vector<int>>{{3,2},{2,3},{1,4}});
    std::vector<std::vector<int>> v = {{3,2},{2,3},{1,4}};
    int n = 5000000;
    int a = 0;
    for(int i = 0; i < n; i++){
        a = v[0][1];
    }
    return 0;
}