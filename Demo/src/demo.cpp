#include "Tenzor.hpp"
#include <iostream>
TZ::mem::salloc* a = new TZ::mem::salloc(1024 * 1024 * 10); // 10MB
TZ::mem::salloc& s = *a;

int main() {
	// int n = 1000;
	//	for (int i = 0; i < n; i++) {
	// int p = rand() % 1024;
	uint8_t* a = static_cast<uint8_t*>(s.allocate(64));
	a[63] = 255;
	s.deallocate((void*&)a, 64);
	//	}
}