#include "Tenzor.hpp"
#include <iostream>

int main() {
	TZ::mem::salloc s(1024 * 1024 * 10); // 1MB
	int n = 1000;
	for (int i = 0; i < n; i++) {
		int p = rand() % 1024;
		uint8_t* a = static_cast<uint8_t*>(s.allocate(p));
		a[0] = 255;
		// s.deallocate((void*&)a, p);
	}
}