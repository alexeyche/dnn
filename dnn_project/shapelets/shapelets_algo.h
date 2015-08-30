#pragma once

#include <deque>
using std::deque;

#include "shapelets_config.h"



namespace dnn {
class TimeSeries;

namespace shapelets {
class Dataset;

class ShapeletInit {
public:
    ShapeletInit();
};


ShapeletInit init;


class ShapeletsAlgo {
public:
    struct AlgoStat : public Printable {
        AlgoStat() : numberOfPruned(0) {}
        void print(std::ostream& str) const {
            str << "AlgoStat:\n";
            str << "\tnumberOfPruned: " << numberOfPruned;
        }

        size_t numberOfPruned;
    };

    ShapeletsAlgo(const ShapeletsConfig &conf);
    ShapeletsAlgo() {}

    void run(Dataset &dataset);

private:
    ShapeletsConfig config;
};




}
}