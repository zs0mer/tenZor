/*#include "nn.hpp"

#include "ExtraFunctions.hpp"


// * Simple demo of a neural network learning XOR

int main() {
    tz::Matrix<float> input({{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}});
    tz::Matrix<float> target({{0.0f}, {1.0f}, {1.0f}, {0.0f}});

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
            diff.apply(output, target[j], zi::SquaredError<float>{}, tz::impl::AnyDevice{});
            loss += diff.sum().copyTo(tz::CPU);
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
#include "grad_tensor.hpp"
#include "math_classes.hpp"
using namespace tz;
using namespace grad;

int main() {
	GradVector<float> a(Vector<float>({2.0f, 3.0f}));
	GradVector<float> b(Vector<float>({5.0f, 4.0f}));
	GradVector<float> c = grad::add<float>(a, b);
	GradVector<float> d(Vector<float>({4.0f, 3.0}));
	GradVector<float> e = grad::multiply<float>(d, c);


	e.backward();

	std::cout << "a: " << a << std::endl;
	std::cout << "b: " << b << std::endl;
	std::cout << "c: " << c << std::endl;
	std::cout << "d: " << d << std::endl;


	return 0;
}
