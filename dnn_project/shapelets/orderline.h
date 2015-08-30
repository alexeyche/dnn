#pragma once

#include <vector>
#include <unordered_set>
#include <unordered_map>

using std::vector;
using std::unordered_set;
using std::unordered_map;

#include <dnn/base/base.h>
#include <dnn/util/log/log.h>

#include "projection.h"
#include "common.h"

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

        unordered_map<size_t, size_t> left_classes_hist;
        unordered_map<size_t, size_t> right_classes_hist;

        size_t N_left;
        size_t N_right;

        unordered_set<size_t> left_major_classes;
        unordered_set<size_t> right_major_classes;

        bool betterThan(const SplitStat& anotherStat) {
            return betterSplitStats(anotherStat.gain, anotherStat.gap, gain, gap);
        }
        // vector<size_t> left_majority;
        // vector<size_t> right_majority;
        void print(std::ostream &o) const {

            o << "SplitStat(gain = " << gain << ", gap = " << gap << ", split_position = " << split_position << ", split_dist = " << split_dist;
            // o << ")";
            o << ", left_hist = ";
            printHist(left_classes_hist, o);
            o << ", right_hist = ";
            printHist(right_classes_hist, o);
            o << ", left_major = ";
            for(const auto& v: left_major_classes) { o << v << ", "; }
            o << ", right_major = ";
            for(const auto& v: right_major_classes) { o << v << ", "; }
        }
        bool leftMajorHas(const size_t &class_id) const {
            return left_major_classes.find(class_id) != left_major_classes.end();
        }
        bool rightMajorHas(const size_t &class_id) const {
            return right_major_classes.find(class_id) != right_major_classes.end();
        }
    };

    Orderline() : _all_min(true) {}

    Orderline& operator=(const Orderline& other);

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
    SplitStat makeBestSplit(const Dataset &d) const;
    const double splitDist(const size_t &split_pos) const;
    double maxPossibleInformationGain(const SplitStat& split_stat, const double &range, const Dataset &d) const;

    static double calcInformationGain(
        const unordered_map<size_t, size_t> &left_hist
      , const unordered_map<size_t, size_t> &right_hist
      , const size_t &N_left
      , const size_t &N_right
      , const double &dataset_entropy
      , const size_t &N
    );
    double maxPossibleFurtherInformationGain(const Dataset &d) const;
private:
    bool _all_min;
    vector<Projection> projs;
};



}
}