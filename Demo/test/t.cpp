#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <Tenzor.hpp>
#include <doctest/doctest.h>

int add(int a, int b) {
	return a + b;
}

TEST_CASE("basic math") {
	CHECK(add(2, 3) == 5);
}