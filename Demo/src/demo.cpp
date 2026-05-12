#include "allocator.hpp"
#include "nn.hpp"

#include "ExtraFunctions.hpp"


// * Simple demo of a neural network learning XOR

int main() {
	tz::Matrix<float> input({{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}});
	tz::Matrix<float> target({{0.0f}, {1.0f}, {1.0f}, {0.0f}});

	input = input.copyTo(tz::GPU);
	target = target.copyTo(tz::GPU);

	zi::NeuralNet<float> nn({2, 32, 32, 32, 1}, tz::GPU,
	                        zi::RrandomUniform<float>{.min = -1.0, .max = 1.0});
	int n = 5000;
	float learnRate = 1.0f;

	for (int i = 0; i <= n; i++) {

		float loss = 0.0f;
		for (uint64_t j = 0; j < target.size(); j++) {
			tz::Vector<float> output = nn.forwardPass(input[j]);
			nn.backwardPass(input[j], target[j]);

			tz::Vector<float> diff;
			tz::impl::TensorIMPL<float>::apply(output.tensor_(), target[j].tensor_(),
			                                   diff.tensor_(), zi::SquaredError<float>{},
			                                   tz::impl::AnyDevice{});
			loss += diff.sum().copyTo(tz::CPU);
		}
		nn.step(learnRate, target.size());

		if (i % 100 == 0) {
			std::cout << "Epoch " << i << ", Loss: " << loss << std::endl;
		}
	}

	return 0;
}
