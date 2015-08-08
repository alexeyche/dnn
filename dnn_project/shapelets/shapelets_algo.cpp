
#include "shapelets_algo.h"

#include <dnn/util/log/log.h>
#include <dnn/util/time_series.h>
#include <dnn/base/factory.h>
#include <dnn/io/stream.h>

#include "dataset.h"
#include "stats.h"
#include "subsequence.h"
#include "projection.h"
#include "dist.h"
#include "orderline.h"

namespace dnn {
namespace shapelets {

ShapeletInit::ShapeletInit() {
    REG_TYPE(ShapeletsConfig);
    REG_TYPE(Subsequence);
}


ShapeletsAlgo::ShapeletsAlgo(const ShapeletsConfig &conf)
: config(conf)
{

}


void ShapeletsAlgo::run(Dataset &dataset) {
    Stats stats(dataset);

    Subsequence best_subsequence;
    Orderline::SplitStat best_split;

    for(size_t k=0; k<dataset.N(); ++k) {
        Ptr<TimeSeries> currentTs = dataset(k);
        L_DEBUG << "Perfoming main loop on dataset " << k;

        stats.calculateSpecificStat(dataset, currentTs);

        for(size_t len = config.startSize ; len <= config.endSize; len += config.stepSize) {
            L_DEBUG << "Finding shapelets with length " << len;
            for(size_t pos = 0; pos < currentTs->length() - len + 1 ; pos++ ) {
                Subsequence candidate(currentTs, k, pos, len);

                // L_DEBUG << "Prepared candidate from " << pos << " to " << pos+len;

                Orderline line(dataset);

                for(size_t l=0; l<dataset.N(); ++l) {
                    double best_dist = std::numeric_limits<double>::max();
                    for(size_t comp_pos=0; comp_pos < (dataset(l)->length()-candidate.length()); ++comp_pos) {
                        Subsequence comp_sub(dataset(l), l, comp_pos, candidate.length());

                        double dist = DistAlgorithm::sdist(candidate, comp_sub, stats);
                        if(dist<best_dist) {
                            best_dist = dist;
                        }
                        if(fabs(best_dist - DistAlgorithm::MIN_DIST) < 1e-10) {
                            break; // Nothing to do here
                        }
                    }

                    line.insert(
                        Projection(l, dataset.getTsLabelId(l), len, best_dist)
                    );
                }
                if(line.illConditioned()) {
                    // L_DEBUG << "Found ill conditioned order line. Ignoring ...";
                    continue;
                }
                Orderline::SplitStat split_stat = line.makeBestSplit();
                // L_INFO << "Comparing: ";
                // L_INFO << best_split;
                // L_INFO << split_stat;
                if(split_stat.betterThan(best_split)) {
                    // L_INFO << "Better!";
                    best_split = split_stat;
                    best_subsequence = candidate;
                }

            }
        }

    }
    L_INFO << "Final best split: " << best_split;
    protobinSave(&best_subsequence, "best.pb");
}


}
}