#pragma once

// # these are changeble:

// # TZ_START_MEM_SIZE : (10 * 1024 * 1024) -> uint64
// # TZ_DEFAULT_ALIGNMENT : (64) -> uint8_t
// # TZ_CTD : (1) -> bool (cross thread deallocation,
// flowing thread deallocation is not supported:
// this is not safe if the two threads are flowing at the same time)
// # TZ_MAX_DIM : (4) -> uint8_t
// # TZ_UNMUTABLE_BRODCASTS : (0) -> bool (cant apply on brodcasted tensors)
// # TZ_ERRORS : (1) -> bool
// # TZ_NORMAL_EQUAL : (1) -> bool
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
