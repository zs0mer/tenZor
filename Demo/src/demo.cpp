#include "TenZor.hpp"
#include <vector>
using namespace tz;
using namespace grad;

int main() {
	// XOR dataset
	std::vector<GradVector<float>> x(4);
	x[0] = GradVector<float>(Vector<float>({0.0f, 0.0f}));
	x[1] = GradVector<float>(Vector<float>({0.0f, 1.0f}));
	x[2] = GradVector<float>(Vector<float>({1.0f, 0.0f}));
	x[3] = GradVector<float>(Vector<float>({1.0f, 1.0f}));

	std::vector<GradVector<float>> target(4);
	target[0] = GradVector<float>(Vector<float>({0.0f}));
	target[1] = GradVector<float>(Vector<float>({1.0f}));
	target[2] = GradVector<float>(Vector<float>({1.0f}));
	target[3] = GradVector<float>(Vector<float>({0.0f}));

	Linear<float> L1(2, 8);
	Linear<float> L2(8, 1);

	Adam<float> optimizer({}, 0.01f, 0.9f, 0.99f);
	optimizer.add(L1.parameters());
	optimizer.add(L2.parameters());

	for (int epoch = 0; epoch <= 1000; epoch++) {
		float total_loss = 0;

		for (int i = 0; i < 4; i++) {
			GradVector<float> h = L1(x[i]);
			GradVector<float> h_act = grad::fn::relu<float>(h);
			GradVector<float> out = L2(h_act);

			GradVector<float> diff = grad::fn::subtract<float>(out, target[i]);
			GradVector<float> sq = grad::fn::multiply<float>(diff, diff);
			GradScalar<float> loss = grad::fn::sum<float>(sq);

			loss.backward();

			total_loss += loss.val();
		}
		optimizer.step(4);

		if (epoch % 100 == 0)
			std::cout << "epoch " << epoch << " loss: " << total_loss / 4 << std::endl;
	}

	return 0;
}
