#pragma once

namespace TZ {

template <class T>
void Tensor<T>::ComputeStrides() {}

template <class T>
template <class nestedVector>
void Tensor<T>::getSTDVecShape(const nestedVector& v, std::vector<uint64_t>& shape) {
	shape.push_back(v.size());

	if constexpr (!isSTDVector<typename nestedVector::value_type>)
		return;
	if (v.empty())
		return;

	const auto& first = v.front();
	for (const auto& sub : v)
		_CHECK(sub.size() != first.size(), "non-rectangular nested vector construction");

	getSTDVecShape(first, shape);
}

template <class T>
template <class nestedVector>
void Tensor<T>::falttenSTDVec(const nestedVector& v, T* dst, uint64_t& offset) {
	if constexpr (isSTDVector<typename nestedVector::value_type>)
		for (const auto& sub : v)
			falttenSTDVec(sub, dst, offset);
	else
		for (const auto& x : v)
			dst[offset++] = x;
}

template <class T>
template <class V>
auto Tensor<T>::ilistToSTDVector(std::initializer_list<V> list) {
	if constexpr (!isIlist<V>)
		return std::vector<V>(list);

	_CHECK(list.size() == 0, "Tensor initializer_list cannot be empty");
	std::vector<decltype(ilistToSTDVector(*list.begin()))> out;
	out.reserve(list.size());

	for (const auto& sub : list)
		out.push_back(ilistToSTDVector(sub));

	return out;
}
}; // namespace TZ