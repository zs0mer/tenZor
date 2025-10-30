#include <vector>



namespace TZ {

    /*
    
    ##############################
    ######## Tensor class ########
    ##############################
    
    */

    template<typename T>
    class tensor
    {
    private:

        // the dimenson of the tensor
        // if 0 the object functions as a scalar
        int dimenson; 

        // the shape of the tensor
        std::vector<int> shape;

        // the stored data
        // using a singel dimenson
        // size = shape[0] * shape[1] ... shape[n]
        // data[i][j]...[k] = data[i * (shape[1] * shape[2] ... shape[n]) + j * (shape[2] * shape[3] ... shape[n]) ... + k]   (I think)
        std::vector<T> data;

        public:

        // standard set with flat std::vector
        void set(const int& newDimenson, const std::vector<int> &newShape, const std::vector<T> &newData);

        // set with multiple vectors nested
        template <typename NestedVector>
        void set(const NestedVector &newData);

        tensor(const int &newDimenson, const std::vector<int> &newShape, const std::vector<T> &newData);
        ~tensor();
    };
    
};