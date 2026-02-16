#pragma once

#define ERRORS 1

#define START_MEM_SIZE 10 * 1024 * 1024
#define DEFAULT_ALIGNMENT 64
#define MAX_DIM 64

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
#include "seters.hpp"
#include "geters.hpp"
#include "metadata_geters.hpp"
#include "tensor_private.hpp"