#pragma once

#include <vector>

using std::vector;


#include <dnn/util/ptr.h>

namespace dnn {
class DoubleMatrix;
class TimeSeries;

namespace shapelets {
class Dataset;


class Stats {
public:
    Stats(const Dataset &_ds);
    ~Stats();
    void calculateSpecificStat(const Dataset &ds, Ptr<TimeSeries> currentTs);
    static void computeDistStat(const TimeSeries &ts_left, const TimeSeries &ts_right, vector<Ptr<DoubleMatrix>> &dst);
    void clean();
    const vector<double>& cumulativeSum(const size_t &ts_id, const size_t &dim) const;
    const vector<double>& squaredCumulativeSum(const size_t &ts_id, const size_t &dim) const;
    const Ptr<DoubleMatrix>& meanProdMatrix(const size_t &ts_id, const size_t &dim) const;
private:
    const Dataset &ds;

    vector<vector<vector<double>>> cumulativeSums;
    vector<vector<vector<double>>> squaredCumulativeSums;

    vector<vector<Ptr<DoubleMatrix>>> prodStat;
};


}
}