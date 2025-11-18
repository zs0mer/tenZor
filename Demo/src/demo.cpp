#include "Tenzor.hpp"


int main(){
    TZ::Tensor<int32_t> t(std::vector<std::vector<int>>{{3,2},{2,3},{1,4}});
    std::vector<std::vector<int>> v = {{1,2},{3,4}};
    long long n = 300000000;
    long long k = 1;
    std::vector<std::vector<int>> a = {{0,0},{1,1}};
    
    for(int i = 0; i< n; i++)
        //k += v[a[0]+i%2][a[1]+i%2];
        k += t[a[i%2]];
    
    std::cout << k;
    return 0;
}
/*#include <vector>
#include <iostream>
#include <stdexcept>

template<typename T>
class Tensor {
    std::vector<T> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t dimension;

    void computeStrides() {
        strides.resize(dimension);
        if (dimension == 0) return;
        strides.back() = 1;
        for (int i = dimension-2; i >= 0; --i)
            strides[i] = strides[i+1] * shape[i+1];
    }

public:
    // Constructor from shape
    Tensor(const std::vector<size_t>& s) : shape(s), dimension(s.size()) {
        size_t total = 1;
        for (auto x : shape) total *= x;
        data.resize(total);
        computeStrides();
    }

    // Lightweight proxy class
    class Slice {
        T* ptr;
        size_t len;
        size_t stride;
        size_t dim_left;
        std::vector<size_t>& strides;
    public:
        Slice(T* p, size_t l, size_t s, size_t d_left, std::vector<size_t>& str)
            : ptr(p), len(l), stride(s), dim_left(d_left), strides(str) {}

        // Return next slice
        Slice operator[](size_t i) {
            if (i >= len) throw std::out_of_range("Index out of range");
            if (dim_left == 1) return Slice(ptr + i*stride, 1, 1, 1, strides);
            return Slice(ptr + i*stride, strides[strides.size()-dim_left+1], stride, dim_left-1, strides);
        }

        // Return reference to the actual data
        operator T&() { return *ptr; }
        T& operator*() { return *ptr; }
    };

    // top-level operator[]
    Slice operator[](size_t i) {
        if (dimension == 0) throw std::runtime_error("Scalar Tensor has no operator[]");
        if (dimension == 1) return Slice(data.data() + i, 1, 1, 1, strides);
        return Slice(data.data() + i*strides[0], strides[1], strides[0], dimension-1, strides);
    }

    // Access flat data directly
    T& atFlat(size_t idx) { return data[idx]; }

    size_t size() const { return data.size(); }
};

int main() {

    Tensor<int32_t> t({3,2});
    std::vector<std::vector<int>> v = {{3,2},{2,3},{1,4}};
    int n = 5000000;
    long long k = 1;
    std::vector<int> a = {2,1};
    
    for(int i = 0; i< n; i++)
        k += t[1][2];
    
    std::cout << k;
    return 0;
}
*/