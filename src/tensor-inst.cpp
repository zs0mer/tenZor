#include "Tenzor.hpp"

namespace TZ {
template class Tensor<int8_t>;
template class Tensor<int16_t>;
template class Tensor<int32_t>;
template class Tensor<int64_t>;
template class Tensor<u_int8_t>;
template class Tensor<u_int16_t>;
template class Tensor<u_int32_t>;
template class Tensor<u_int64_t>;
template class Tensor<float>;
template class Tensor<double>;


// int8_t
template Tensor<int8_t>::Tensor(const int &, const std::vector<int> &, const std::vector<int8_t> &);
template void Tensor<int8_t>::set(const int &, const std::vector<int> &, const std::vector<int8_t> &);;
template int Tensor<int8_t>::getDimenson() const;;
template std::vector<int> Tensor<int8_t>::getShape() const;;
template void Tensor<int8_t>::printShape(const std::string &) const;;


// int16_t
template Tensor<int16_t>::Tensor(const int &, const std::vector<int> &, const std::vector<int16_t> &);
template void Tensor<int16_t>::set(const int &, const std::vector<int> &, const std::vector<int16_t> &);;
template int Tensor<int16_t>::getDimenson() const;;
template std::vector<int> Tensor<int16_t>::getShape() const;;
template void Tensor<int16_t>::printShape(const std::string &) const;;


// int32_t
template Tensor<int32_t>::Tensor(const int &, const std::vector<int> &, const std::vector<int32_t> &);
template void Tensor<int32_t>::set(const int &, const std::vector<int> &, const std::vector<int32_t> &);;
template int Tensor<int32_t>::getDimenson() const;;
template std::vector<int> Tensor<int32_t>::getShape() const;;
template void Tensor<int32_t>::printShape(const std::string &) const;;


// int64_t
template Tensor<int64_t>::Tensor(const int &, const std::vector<int> &, const std::vector<int64_t> &);
template void Tensor<int64_t>::set(const int &, const std::vector<int> &, const std::vector<int64_t> &);;
template int Tensor<int64_t>::getDimenson() const;;
template std::vector<int> Tensor<int64_t>::getShape() const;;
template void Tensor<int64_t>::printShape(const std::string &) const;;


// u_int8_t
template Tensor<u_int8_t>::Tensor(const int &, const std::vector<int> &, const std::vector<u_int8_t> &);
template void Tensor<u_int8_t>::set(const int &, const std::vector<int> &, const std::vector<u_int8_t> &);;
template int Tensor<u_int8_t>::getDimenson() const;;
template std::vector<int> Tensor<u_int8_t>::getShape() const;;
template void Tensor<u_int8_t>::printShape(const std::string &) const;;


// u_int16_t
template Tensor<u_int16_t>::Tensor(const int &, const std::vector<int> &, const std::vector<u_int16_t> &);
template void Tensor<u_int16_t>::set(const int &, const std::vector<int> &, const std::vector<u_int16_t> &);;
template int Tensor<u_int16_t>::getDimenson() const;;
template std::vector<int> Tensor<u_int16_t>::getShape() const;;
template void Tensor<u_int16_t>::printShape(const std::string &) const;;


// u_int32_t
template Tensor<u_int32_t>::Tensor(const int &, const std::vector<int> &, const std::vector<u_int32_t> &);
template void Tensor<u_int32_t>::set(const int &, const std::vector<int> &, const std::vector<u_int32_t> &);;
template int Tensor<u_int32_t>::getDimenson() const;;
template std::vector<int> Tensor<u_int32_t>::getShape() const;;
template void Tensor<u_int32_t>::printShape(const std::string &) const;;


// u_int64_t
template Tensor<u_int64_t>::Tensor(const int &, const std::vector<int> &, const std::vector<u_int64_t> &);
template void Tensor<u_int64_t>::set(const int &, const std::vector<int> &, const std::vector<u_int64_t> &);;
template int Tensor<u_int64_t>::getDimenson() const;;
template std::vector<int> Tensor<u_int64_t>::getShape() const;;
template void Tensor<u_int64_t>::printShape(const std::string &) const;;


// float
template Tensor<float>::Tensor(const int &, const std::vector<int> &, const std::vector<float> &);
template void Tensor<float>::set(const int &, const std::vector<int> &, const std::vector<float> &);;
template int Tensor<float>::getDimenson() const;;
template std::vector<int> Tensor<float>::getShape() const;;
template void Tensor<float>::printShape(const std::string &) const;;


// double
template Tensor<double>::Tensor(const int &, const std::vector<int> &, const std::vector<double> &);
template void Tensor<double>::set(const int &, const std::vector<int> &, const std::vector<double> &);;
template int Tensor<double>::getDimenson() const;;
template std::vector<int> Tensor<double>::getShape() const;;
template void Tensor<double>::printShape(const std::string &) const;;


};
