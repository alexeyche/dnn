#pragma once

#include <stddef.h>

namespace dnn {
class DoubleMatrix;

namespace shapelets {
class Subsequence;
class Stats;


class DistAlgorithm {
public:
    static double mean(const Subsequence& sub, const Stats &stats, const size_t &dim);
    static double meanSquared(const Subsequence& sub, const Stats &stats, const size_t &dim);
    static double meanProd(const Subsequence& left_sub, const Subsequence& right_sub, const Stats &stats, const size_t &dim);
    static double sdist(const Subsequence &left, const Subsequence &right, const Stats &stats);

    static const double MIN_DIST;
};



}
}