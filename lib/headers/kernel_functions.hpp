#pragma once

#include <cstdint>

void* GPUAlloc(const uint64_t bytes, const uint8_t alignment);

void GPUFree(void* ptr, const uint64_t bytes);
