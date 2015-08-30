#pragma once

#include <unordered_map>
#include <map>

using std::map;
using std::unordered_map;

namespace dnn {
namespace shapelets {

template <typename T>
void printHist(const unordered_map<size_t, size_t> &h, T &&o) {
    map<size_t, size_t> oh(h.begin(), h.end());
    size_t i=0;
    for(const auto &v: oh) {
        o << v.second;
        ++i;
        if(i < oh.size()) o << ", ";
    }
}




}
}