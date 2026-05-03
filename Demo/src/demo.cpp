#include "nn.hpp"
#include <Tenzor.hpp>

using namespace tz;

int main() {
	zi::NeuralNet<float> nn({2, 3, 1});

	tz::Vector<float> input({0.5f, 0.8f});
	tz::Vector<float> target({0.3f});

	tz::Vector<float> output = nn.forwardPass(input);
	std::cout << "Output: " << output << std::endl;

	nn.backwardPass(input, target);
	nn.step(0.01f, 1);
	return 0;
}
