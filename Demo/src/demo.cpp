/*#include "nn.hpp"

#include "ExtraFunctions.hpp"


// * Simple demo of a neural network learning XOR

int main() {
    tz::Matrix<float> input({{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f},
{1.0f, 1.0f}}); tz::Matrix<float> target({{0.0f}, {1.0f}, {1.0f}, {0.0f}});

    input = input.copyTo(tz::CPU);
    target = target.copyTo(tz::CPU);

    zi::NeuralNet<float> nn({2, 32, 32, 32, 1}, tz::CPU,
                            zi::RrandomUniform<float>{.min = -1.0, .max = 1.0});
    int n = 5000;
    float learnRate = 1.0f;

    for (int i = 0; i <= n; i++) {

        float loss = 0.0f;
        for (uint64_t j = 0; j < target.size(); j++) {
            tz::Vector<float> output = nn.forwardPass(input[j]);
            nn.backwardPass(input[j], target[j]);

            tz::Vector<float> diff;
            diff.apply(output, target[j], zi::SquaredError<float>{},
tz::impl::AnyDevice{}); loss += diff.sum().copyTo(tz::CPU);
        }
        nn.step(learnRate, target.size());

        if (i % 100 == 0) {
            std::cout << "Epoch " << i << ", Loss: " << loss << std::endl;
        }
    }

    return 0;
}
*/

#include "TenZor.hpp"
using namespace tz;
using namespace grad;

int main() {
	// XOR dataset — 4 samples, 2 features
	GradMatrix<float> x(Matrix<float>(4, 2, CPU));
	x.val().at(0, 0) = 0;
	x.val().at(0, 1) = 0;
	x.val().at(1, 0) = 0;
	x.val().at(1, 1) = 1;
	x.val().at(2, 0) = 1;
	x.val().at(2, 1) = 0;
	x.val().at(3, 0) = 1;
	x.val().at(3, 1) = 1;

	// targets
	GradVector<float> target(Vector<float>(4, CPU));
	target.val().at(0) = 0;
	target.val().at(1) = 1;
	target.val().at(2) = 1;
	target.val().at(3) = 0;

	// weights: 2 -> 4 -> 1
	GradMatrix<float> w1(Matrix<float>(2, 4, CPU));
	GradVector<float> w2(Vector<float>(4, CPU));

	// random init
	w1.val().apply([](float& x) { x = (float)rand() / RAND_MAX * 0.2f - 0.1f; });
	w2.val().apply([](float& x) { x = (float)rand() / RAND_MAX * 0.2f - 0.1f; });

	float lr = 0.1f;

	for (int epoch = 0; epoch < 1000; epoch++) {
		// forward: (4x2) * (2x4) -> (4x4), then matvec (4x4) * (4) -> (4)
		GradMatrix<float> h = grad::matmul<float>(x, w1);
		GradMatrix<float> h_act = grad::relu<float>(h);
		GradVector<float> out = grad::matmul<float>(h_act, w2);

		// MSE loss
		GradVector<float> diff = grad::subtract<float>(out, target);
		GradVector<float> sq = grad::multiply<float>(diff, diff);
		GradScalar<float> loss = grad::sum<float>(sq);

		// backward
		loss.backward();

		// gradient descent step
		w1.val() -= w1.grad() * Scalar<float>(lr);
		w2.val() -= w2.grad() * Scalar<float>(lr);

		// zero grads
		w1.grad().setAll(0);
		w2.grad().setAll(0);

		if (epoch % 100 == 0)
			std::cout << "epoch " << epoch << " loss: " << loss << std::endl;
	}

	return 0;
}
