#include "stats.h"

#include <dnn/util/matrix.h>
#include <dnn/util/time_series.h>

#include "dataset.h"

namespace dnn {
namespace shapelets {

Stats::Stats(const Dataset &_ds) : ds(_ds) {
    L_DEBUG << "Calculating independent time series related stastics";
    for(const auto &ts: ds) {
        vector<vector<double>> cumSums;
        vector<vector<double>> cumSumsSquared;
        for(size_t di=0; di<ts->dim(); ++di) {
            auto vec = ts->getVector(di);

            vector<double> v(vec.size()+1);
            vector<double> v_sq(vec.size()+1);

            v[0] = 0.0;
            v_sq[0] = 0.0;

            for(size_t k=0; k<vec.size(); ++k) {
                v[k+1] = v[k] + vec[k];
                v_sq[k+1] = v_sq[k] + vec[k]*vec[k];
            }

            cumSums.push_back(v);
            cumSumsSquared.push_back(v_sq);
        }
        cumulativeSums.push_back(cumSums);
        squaredCumulativeSums.push_back(cumSumsSquared);
    }
    L_DEBUG << "Done";
}
Stats::~Stats() {
    clean();
}

void Stats::clean() {
    if(prodStat.size()>0) {
        for(auto ts_m: prodStat) {
            for(auto dim_m: ts_m) {
                delete dim_m.ptr();
            }
            ts_m.clear();
        }
        prodStat.clear();
    }
}

void Stats::calculateSpecificStat(const Dataset &ds, Ptr<TimeSeries> currentTs) {
    clean();
    L_DEBUG << "Calculating dependent time series related stastics";
    for(size_t ni=0; ni<ds.N(); ++ni) {
        if(currentTs->dim() != ds(ni)->dim()) {
            throw dnnException() << "Found non homogeneous dataset while calculating stat";
        }

        vector<Ptr<DoubleMatrix>> tsStat;
        for(size_t di=0; di<currentTs->dim(); ++di) {
            tsStat.push_back(new DoubleMatrix());
        }
        prodStat.push_back(tsStat);
        computeDistStat(currentTs.ref(), ds(ni).ref(), prodStat.back());
    }
    L_DEBUG << "Done";
}

const vector<double>& Stats::cumulativeSum(const size_t &ts_id, const size_t &dim) const {
    return cumulativeSums[ts_id][dim];
}
const vector<double>& Stats::squaredCumulativeSum(const size_t &ts_id, const size_t &dim) const {
    return squaredCumulativeSums[ts_id][dim];
}

const Ptr<DoubleMatrix>& Stats::meanProdMatrix(const size_t &ts_id, const size_t &dim) const {
    return prodStat[ts_id][dim];
}

void Stats::computeDistStat(
    const TimeSeries &ts_left
  , const TimeSeries &ts_right
  , vector<Ptr<DoubleMatrix>> &dst
) {
    if(dst.size() != ts_left.dim()) {
        throw dnnException() << "Left time series has wrong dimenstion";
    }
    if(dst.size() != ts_right.dim()) {
        throw dnnException() << "Right time series has wrong dimenstion";
    }
    size_t n = ts_left.length();
    size_t m = ts_right.length();

    for(size_t di=0; di<dst.size(); ++di) {
        const vector<double> &x = ts_left.getVector(di);
        const vector<double> &y = ts_right.getVector(di);
        DoubleMatrix &d = dst[di].ref();
        d.allocate(n+1, m+1);

        for(size_t col = 0; col < ts_left.length(); ++col) {
            for(size_t row = 0; row < ts_right.length(); ++row) {
                if( (col == 0) || (row == 0) ) {
                    d(col, row) = 0.0;
                    continue;
                }
                d(col, row) = d(col-1, row-1) + x[col-1] * y[row - 1];
            }
        }

    }


    // size_t L = (n > m) ? n : m;

    // for(size_t di=0; di<dst.size(); ++di) {
    //     const vector<double> &x = ts_left.getVector(di);
    //     const vector<double> &y = ts_right.getVector(di);
        // DoubleMatrix &d = dst[di].ref();
        // d.allocate(n+1, m+1);

    //     int i , j , k;
    //     for( k = 0 ; k < L ; k++ )
    //     {
    //         // L_DEBUG << "k: " << k;
    //         if( k <= n )
    //         {
    //             d(k,0) = 0.0; //x[k]*y[0];
    //             for( i = k+1 , j = 0 ; i < n && j < m ; i++, j++ ) {
    //                 // L_DEBUG << "i: " << i;
    //                 d(i+1, j) = d(i-1, j-1)+x[i]*y[j];
    //             }
    //         }

    //         if( k <= m )
    //         {
    //             d(0, k) = 0.0; //x[0]*y[k];
    //             for( i = 1 , j = k+0 ; i < n && j < m ; i++, j++ ) {
    //                d(i, j) = d(i-1, j-1)+x[i]*y[j];
    //                // L_DEBUG << "i: " << i;
    //            }
    //         }
    //     }
    //     // throw dnnException() << "OUT";
    // }
}



}
}