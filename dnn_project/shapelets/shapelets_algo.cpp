
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

struct Split {
    Split() {}
    Split(
        const Subsequence &_subsequence
      , const Orderline &_orderline
      , const Orderline::SplitStat &_stat
    ) :
      subsequence(_subsequence)
    , orderline(_orderline)
    , stat(_stat)
    {
    }

    Subsequence subsequence;
    Orderline orderline;
    Orderline::SplitStat stat;
};



void ShapeletsAlgo::run(Dataset &dataset) {
    AlgoStat algo_stat;
    Stats stats(dataset);

    Split best_split;

    for(size_t k=0; k<dataset.N(); ++k) {
        Ptr<TimeSeries> currentTs = dataset(k);
        L_DEBUG << "Perfoming main loop on dataset " << k;

        stats.calculateSpecificStat(dataset, currentTs);

        for(size_t len = config.startSize ; len <= config.endSize; len += config.stepSize) {
            L_DEBUG << "Finding shapelets with length " << len;
            deque<Split> poorCache;

            for(size_t pos = 0; pos < currentTs->length() - len + 1 ; pos++ ) {
                Subsequence candidate(currentTs, k, pos, len);

                // L_DEBUG << "Prepared candidate from " << pos << " to " << pos+len;
                // protobinSave(&candidate, "candidate.pb");

                Orderline line;

                bool pruned = false;
                double pruned_ig = 0;
                for(auto p = poorCache.end()-1; p>=poorCache.begin(); --p) {
                    double dist = DistAlgorithm::sdist(candidate, p->subsequence, stats);
                    if(dist>0.05) continue;

                    // L_INFO << "Got dist with poor, R == " << dist << ", gain: " << p->stat.gain << " he has orderline:";
                    // L_INFO << p->orderline;
                    // L_INFO << "and split_stat: ";
                    // L_INFO << p->stat;
                    double max_ig = p->orderline.maxPossibleInformationGain(p->stat, dist, dataset);
                    if(max_ig < best_split.stat.gain) {
                        L_INFO << "Found max informational gain: " << max_ig;
                        pruned = true;
                        // pruned_ig = max_ig;
                        break;
                    }
                }
                if(pruned) {
                    L_INFO << "Pruning ...";
                    algo_stat.numberOfPruned++;
                    // continue;
                }

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

                    if(!line.illConditioned()) {
                        double max_further_ig = line.maxPossibleFurtherInformationGain(dataset);
                        L_INFO << "Got max further IG: " << max_further_ig;
                        if(max_further_ig < best_split.stat.gain) {
                            pruned_ig = max_further_ig;
                            pruned = true;
                            L_INFO << "Got further IG " << max_further_ig << " less than max gain " << best_split.stat.gain;
                        }
                    }
                }
                if(line.illConditioned()) {
                    // L_DEBUG << "Found ill conditioned order line. Ignoring ...";
                    continue;
                }
                Orderline::SplitStat split_stat = line.makeBestSplit(dataset);
                // L_INFO << "Got orderline: " << line;
                L_INFO << "Comparing, the best, and candidate:";
                L_INFO << best_split.stat;
                L_INFO << split_stat;

                // for(const auto& v: max_possibles) {
                //     if(v < split_stat.gain) {
                //         L_INFO << "Algo sucks: " << v << " < " << split_stat.gain;
                //     } else {
                //         L_INFO << "Algo rulezz: " << v << " >= " << split_stat.gain;
                //     }
                // }

                if(pruned && (pruned_ig<split_stat.gain)) {
                    throw dnnException() << "Fuckup\n";
                }
                if(split_stat.betterThan(best_split.stat)) {
                    // L_INFO << "Better!";
                    best_split = Split(candidate, line, split_stat);
                    L_INFO << "Found better orderline:";
                    L_INFO << best_split.orderline;
                } else {
                    if(poorCache.size()>config.poorCacheSize) {
                        poorCache.pop_front();
                    }
                    poorCache.emplace_back(candidate, line, split_stat);
                }

            }
        }

    }
    L_INFO << algo_stat;

    L_INFO << "Final best split: " << best_split.stat;
    L_INFO << "Final best split: " << best_split.orderline;
    protobinSave(&best_split.subsequence, "best.pb");
}


}
}