#include <vector>
namespace TZ {

    /*
    
    ##############################
    ######## Tensor class ########
    ##############################
    
    */

    template<class T>
    class tensor
    {
    private:

        // the dimenson of the tensor
        // if 0 the object functions as a scalar
        int dimenson = 0; 

        // the shape of the tensor
        std::vector<int> shape = {};

        // the stored data
        // using a singel dimenson
        // size = shape[0] * shape[1] * ... * shape[n]
        std::vector<T> data;

        public:
        tensor(/* args */);
        ~tensor();
    };
    
};