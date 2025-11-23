#include "Tenzor.hpp"

TZ::mem::salloc saloc(5 * 1024 * 1024);

int main() {
	void* a = saloc.allocate(10);
}