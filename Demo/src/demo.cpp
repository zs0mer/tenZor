#include "TenZor.hpp"
#include <iostream>
#include <cmath>

using namespace tz;
using namespace grad;

static GradVector<float> makeVec(std::initializer_list<float> vals, Device dev) {
	Vector<float> v(vals.size(), CPU); // create on CPU to fill with at()
	uint64_t i = 0;
	for (float f : vals)
		v.at(i++) = f;
	return GradVector<float>(v.copyTo(dev));
}
int main() {
	const Device dev = GPU;

	GradVector<float> x[4] = {
	    makeVec({0.f, 0.f}, dev),
	    makeVec({0.f, 1.f}, dev),
	    makeVec({1.f, 0.f}, dev),
	    makeVec({1.f, 1.f}, dev),
	};
	GradVector<float> y[4] = {
	    makeVec({0.f}, dev),
	    makeVec({1.f}, dev),
	    makeVec({1.f}, dev),
	    makeVec({0.f}, dev),
	};

	// network: 2 -> 1024 -> 1024 -> 1
	Linear<float> L1(2, 8, dev);
	Linear<float> L2(8, 8, dev);
	Linear<float> L3(8, 1, dev);

	SGD<float> opt({}, 0.5f);
	opt.add(L1.parameters());
	opt.add(L2.parameters());
	opt.add(L3.parameters());

	for (int epoch = 1; epoch <= 5000; epoch++) {
		float total_loss = 0.f;

		for (int i = 0; i < 4; i++) {
			auto z1 = L1(x[i]);
			auto h = fn::sigmoid<float, GradVector<float>>(z1);
			auto z2 = L2(h);
			auto h2 = fn::sigmoid<float, GradVector<float>>(z2);
			auto z3 = L3(h2);
			auto out = fn::sigmoid<float, GradVector<float>>(z3);
			auto diff = fn::subtract<float, GradVector<float>>(out, y[i]);
			auto sq = fn::multiply<float, GradVector<float>>(diff, diff);
			auto loss = fn::sum<float>(sq);

			total_loss += loss.val().copyTo(CPU).get();
			loss.backward();
		}

		opt.step(4);

		if (epoch % 500 == 0)
			std::cout << "epoch " << epoch << "  loss: " << total_loss / 4.f << "\n";
	}

	std::cout << "\nResults:\n";
	for (int i = 0; i < 4; i++) {
		auto z1 = L1(x[i]);
		auto h = fn::sigmoid<float, GradVector<float>>(z1);
		auto z2 = L2(h);
		auto out = fn::sigmoid<float, GradVector<float>>(z2);

		float pred = out.val().copyTo(CPU).at(0);
		float expected = y[i].val().copyTo(CPU).at(0);

		std::cout << "pred: " << pred << "  expected: " << expected
		          << (std::round(pred) == expected ? "  ✓" : "  ✗") << "\n";
	}

	return 0;
}
