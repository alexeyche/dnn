#pragma once

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
    ShapeletsAlgo(const ShapeletsConfig &conf);
    ShapeletsAlgo() {}

    void run(Dataset &dataset);

private:
    ShapeletsConfig config;
};


}
}