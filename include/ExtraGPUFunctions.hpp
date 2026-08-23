
// # you can put here any extra functions that you need for GPU operations
// # for example:
// template <class T>
// struct Add {
// 	  TZ_HOST_DEVICE void operator()(const T& a, const T& b, T& c) const {
// 		 c = a + b;
// 	  }
// };
// # you have to use TZ_HOST_DEVICE to make it work
// # you also have to use a functor struct

// # you have to instantiate the functions
// # for example:
// define TZ_EXTRA_GPU_FUNCTIONS_INSTANTIATE                                                   \
// INSTANTIATE_TERNARY_APPLY(T, Add);

// # you can do:
// INSTANTIATE_UNARY_APPLY(T, FUNC);
// INSTANTIATE_BINARY_APPLY(T, FUNC);
// INSTANTIATE_TERNARY_APPLY(T, FUNC);
//
// INSTANTIATE_UNARY_APPLY(T, Func):
//	template void applyGPU<T, Func<T>>(SimpleTensor<T>, Func<T>)
//
// INSTANTIATE_BINARY_APPLY(T, Func):
//	template void applyGPU<T, Func<T>>(const SimpleTensor<T>, SimpleTensor<T>, Func<T>)
//
// INSTANTIATE_TERNARY_APPLY(T, Func):
//	template void applyGPU<T, Func<T>>(const SimpleTensor<T>, const SimpleTensor<T>,
//	                                   SimpleTensor<T>, Func<T>)

// # make sure to include a header
