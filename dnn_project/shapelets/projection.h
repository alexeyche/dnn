#pragma once

#include <limits>

namespace dnn {
namespace shapelets {

struct Projection {
    Projection(size_t _ts_id, size_t _class_id, size_t _length)
    : ts_id(_ts_id)
    , class_id(_class_id)
    , length(_length)
    , dist(std::numeric_limits<double>::max())
    {
    }

    size_t ts_id;
    size_t class_id;

    size_t length;
    double dist;
};


}
}