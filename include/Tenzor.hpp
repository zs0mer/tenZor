#pragma once

// # these are changeable:

// # TZ_START_MEM_SIZE : (10 * 1024 * 1024) -> uint64
// # TZ_DEFAULT_ALIGNMENT : (64) -> uint8_t
// # TZ_CTD : (1) -> bool (cross thread deallocation,
// concurrent thread deallocation is not supported:
// both threads must not run simultaneously)
// # TZ_MAX_DIM : (4) -> uint8_t
// # TZ_IMMUTABLE_BROADCASTS : (1) -> bool (can't apply on broadcasted tensors)
// # TZ_ERRORS : (1) -> bool
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
