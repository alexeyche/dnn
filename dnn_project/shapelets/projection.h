#pragma once

#include <stddef.h>
#include <limits>

namespace dnn {
namespace shapelets {

struct Projection {
    Projection(size_t _ts_id, size_t _class_id, size_t _length, double _dist)
    : ts_id(_ts_id)
    , class_id(_class_id)
    , length(_length)
    , dist(_dist)
    {
    }

    size_t ts_id;
    size_t class_id;

    size_t length;
    double dist;
};


}
}