#pragma once

#include <vector>

using std::vector;

#include <dnn/base/base.h>
#include <dnn/util/log/log.h>

#include "projection.h"

namespace dnn {
namespace shapelets {

class Dataset;


class Orderline : public Printable {
public:
    static bool betterSplitStats(const double &gainBase, const double &gapBase, const double &gainNew, const double &gapNew) {
        // L_DEBUG << "Comparing: " << gainBase << " " << gapBase << " and " << gainNew << " " << gapNew;
        if( (gainNew > gainBase ) ||
            ((gainNew == gainBase) && (gapNew > gapBase)) ) {
            // L_DEBUG << "true";
            return true;
        }
        // L_DEBUG << "false";
        return false;
    }

    struct SplitStat : public Printable {
        SplitStat() : gain(0.0), gap(0.0), split_dist(0.0), split_position(std::numeric_limits<size_t>::max()) {}
        double gain;
        double gap;
        size_t split_position;
        double split_dist;

        bool betterThan(const SplitStat& anotherStat) {
            return betterSplitStats(anotherStat.gain, anotherStat.gap, gain, gap);
        }
        // vector<size_t> left_majority;
        // vector<size_t> right_majority;
        void print(std::ostream &o) const {
            o << "SplitStat(gain = " << gain << ", gap = " << gap << ", split_position = " << split_position << ", split_dist = " << split_dist << ")";
        }
    };

    Orderline(const Dataset &_d) : d(_d), _all_min(true) {}
    void insert(Projection &&newp);

    void print(std::ostream &o) const {
        o << "Orderline, " << projs.size() << " projections\n";
        for(const auto &p: projs) {
            o << " |" << p.class_id << ", " << p.dist << "| ";
        }
        o << "\n";
    }
    const bool& illConditioned() const {
        return _all_min;
    }
    SplitStat makeBestSplit() const;
    const double splitDist(const size_t &split_pos) const;
private:
    bool _all_min;
    const Dataset &d;
    vector<Projection> projs;
};



}
}