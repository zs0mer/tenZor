#include "Catch2/catch.hpp"
#include "Tenzor.hpp"
#include <iostream>

TZ::mem::salloc saloc(5 * 1024 * 1024);

int main() {
	char* a = (char*)saloc.allocate(2 * 1024);
	// int* b = new int;
	std::cout << a[1];
}