
#include "option_parser.h"

#include <dnn/util/accum.h>

namespace dnn {

template <>
double cast(string &s) {
	return std::stof(s);
}

template <>
int cast(string &s) {
	return std::stoi(s);
}

template <>
size_t cast(string &s) {
	int i = std::stoi(s);
	if(i<0) {
		throw dnnException() << "Can't cast signed value to unsigned: " << s << "\n";
	}
	return static_cast<size_t>(i);
}
template <>
string cast(string &s) {
	return string(s);
}


template <>
Accum<string> cast(string &s) {
    return Accum<string>(s);
}


}