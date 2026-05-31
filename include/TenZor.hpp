#pragma once

// # these are changeable:

// # TZ_CUDA_AVAILABLE : (1) -> bool
// # TZ_START_MEM_SIZE : (10 * 1024 * 1024) -> uint64
// # TZ_DEFAULT_ALIGNMENT : (64) -> uint8_t
// # TZ_CTD : (1) -> bool (cross thread deallocation,
// concurrent thread deallocation is not supported:
// both threads must not run simultaneously)
// # TZ_MAX_DIM : (4) -> uint8_t
// # TZ_IMMUTABLE_BROADCASTS : (1) -> bool (can't apply on broadcasted tensors)
// # TZ_ERRORS : (1) -> bool
// # TZ_APPLY_ERROR_IF_SHAPE_NOT_SAME : (1) -> bool
// (if the tensors are not the same in apply,
// the second tensor will be set to the same shape)
// # TZ_DEFAULT_ALLOCATOR :
// (
// 	if (device == CPU)
//		return Salloc::instance();
//
//	if (device == GPU)
//		return Salloc::instance();
//
//	return Malloc::instance();
// )
// # -> "function snippet"

#include "math_classes.hpp"
#include "operators.hpp"
#include "tensor_impl.hpp"
#include "grad_functions.hpp"
#include "grad_extras.hpp"
