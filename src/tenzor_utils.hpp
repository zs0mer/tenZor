#pragma once
#include <string>
#include <iostream>
#include <ctime>

#define CHECK(expresson, error) _check((expresson), (error), __FILE__, __LINE__, __func__)
#define CHECK_(expresson) _check((expresson), "unexpected", __FILE__, __LINE__, __func__)


// a function to make error handleing easier
void _check(const bool expresson, const std::string& error, const char* file, int line, const char* func){
    if(expresson){
        std::time_t now = std::time(nullptr);
        std::string time = std::ctime(&now);
        time.pop_back(); // remove trailing newline

        throw std::runtime_error(
            "Error: " + error +
            "\nFile: " + file +
            "\nLine: " + std::to_string(line) +
            "\nFunction: " + func +
            "\nTime: " + time
        );
    }
}
