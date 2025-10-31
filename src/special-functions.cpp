#include <string>
#include <iostream>
#include <ctime>


// a static function to make error handleing easier
inline static void check_impl(const bool expresson, const std::string& error, const char* file, int line, const char* func){
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