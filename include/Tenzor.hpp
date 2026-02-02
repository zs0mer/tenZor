#pragma once

#define ERRORS 1

#define START_MEM_SIZE 10 * 1024 * 1024

#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <mutex>
#include <cstring>

#include "tenzor_utils.hpp"
#include "allocator.hpp"
#include "buffer.hpp"
#include "tensor.hpp"
