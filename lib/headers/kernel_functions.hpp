#pragma once

#include <cstdint>

void* allocGPU(const uint64_t bytes, const uint8_t alignment);

void freeGPU(void* ptr, const uint64_t bytes);

void copyToGPU(void* to, void* from, const uint64_t bytes);

void copyToCPU(void* to, void* from, const uint64_t bytes);

void memCopyGPU(void* to, void* from, const uint64_t bytes);
