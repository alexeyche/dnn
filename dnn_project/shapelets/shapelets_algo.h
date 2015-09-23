#pragma once

#include <deque>
using std::deque;

#include "shapelets_config.h"



namespace dnn {
class TimeSeries;

namespace shapelets {
class Dataset;

#define REG_FILE <shapelets/shapelet_register.x>
#include <dnn/base/forward_declarations.x>
#undef REG_FILE

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