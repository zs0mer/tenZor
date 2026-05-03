#include "math_classes.hpp"
#include "ExtraFunctions.hpp"
#include "nn.hpp"

using namespace tz;

int main() {
	zi::NeuralNet<float> nn({2, 8, 4, 1}, zi::RrandomUniform<float>{.min = -1.0, .max = 1.0});

	tz::Matrix<float> input({{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}});
	tz::Matrix<float> target({{0.0f}, {1.0f}, {1.0f}, {0.0f}});
	int n = 10000;
	float learnRate = .5;

	for (int i = 0; i <= n; i++) {

		float loss = 0.0f;
		for (uint64_t j = 0; j < target.size(); j++) {
			tz::Vector<float> output = nn.forwardPass(input[j]);
			nn.backwardPass(input[j], target[j]);
			float diff = (output - target[j]).sum();
			loss += (diff * diff) / 2;
		}
		nn.step(learnRate, target.size());

		if (i % 10 == 0) {
			std::cout << "Epoch " << i << ", Loss: " << loss << nn.forwardPass(input[0]) << " ";
			std::cout << nn.forwardPass(input[1]) << " " << nn.forwardPass(input[2]) << " ";
			std::cout << nn.forwardPass(input[3]) << " " << std::endl;
		}
	}


	return 0;
}
