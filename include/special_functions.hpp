#include "Tenzor.hpp"

#define CHECK(expresson, error) _check((expresson), (error), __FILE__, __LINE__, __func__)
#define CHECK_S(expresson) _check((expresson), "unexpected", __FILE__, __LINE__, __func__)


