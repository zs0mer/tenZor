#pragma once

namespace TZ {

template <class T>
Tensor<T>::Tensor(const std::vector<uint64_t>& shape, const mem::Buffer data,
                  const std::vector<uint64_t>& strides, const uint64_t offset)
    : shape_(shape), data_(data), strides_(strides), offset_(offset) {}

template <class T>
void Tensor<T>::ComputeStrides() {
	uint64_t k = 1;
	strides_.resize(shape_.size());

	for (int64_t i = shape_.size(); i-- > 0;) {
		strides_[i] = k;
		k *= shape_[i];
	}
}

template <class T>
template <class NestedVector>
void Tensor<T>::getSTDVecShape(const NestedVector& v, std::vector<uint64_t>& shape) {

	if constexpr (isSTDVector<NestedVector>::value) {
		shape.push_back(v.size());

		if (v.empty())
			return;

		using Inner = typename NestedVector::value_type;

		if constexpr (isSTDVector<Inner>::value) {
			const auto& first = v.front();
			getSTDVecShape(first, shape);
		}
	}
}

template <class T>
template <class NestedVector>
void Tensor<T>::falttenSTDVec(const NestedVector& v, T* dst, uint64_t& offset) {
	if constexpr (isSTDVector<NestedVector>::value)
		for (const auto& sub : v)
			falttenSTDVec(sub, dst, offset);
	else
		dst[offset++] = static_cast<T>(v);
}

template <class T>
template <class L>
auto Tensor<T>::ilistToSTDVector(std::initializer_list<L> list) {
	if constexpr (!isIlist<L>::value)
		return std::vector<L>(list);

	_CHECK(list.size() == 0, "Tensor initializer_list cannot be empty");
	std::vector<decltype(ilistToSTDVector(*list.begin()))> out;
	out.reserve(list.size());

	for (const auto& sub : list)
		out.push_back(ilistToSTDVector(sub));

	return out;
}

}; // namespace TZ