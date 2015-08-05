
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
    for(size_t k=0; k<dataset.N(); ++k) {
        Ptr<TimeSeries> currentTs = dataset(k);
        L_DEBUG << "Perfoming main loop on dataset " << k;

        stats.calculateSpecificStat(dataset, currentTs);

        for(size_t len = config.startSize ; len <= config.endSize; len += config.stepSize) {
            L_DEBUG << "Finding shapelets with length " << len;
            for(size_t pos = 0; pos < currentTs->length() - len + 1 ; pos++ ) {
                Subsequence sub(currentTs, k, pos, len);
                ofstream f("left.pb");
                Stream s(f, Stream::Binary);
                s.writeObject(&sub);

                L_DEBUG << "Prepared candidate from " << pos << " to " << pos+len;
                for(size_t l=0; l<dataset.N(); ++l) {
                    Projection prj(l, dataset(l)->getLabelId(), len);
                    double best_dist = std::numeric_limits<double>::max();
                    for(size_t comp_pos=0; comp_pos < (dataset(l)->length()-sub.length()); ++comp_pos) {
                        Subsequence comp_sub(dataset(l), l, comp_pos, sub.length());
                        stringstream ss;
                        ss << "right" << comp_pos << ".pb";
                        ofstream f2(ss.str());
                        Stream s2(f2, Stream::Binary);
                        s2.writeObject(&comp_sub);

                        double dist = DistAlgorithm::sdist(sub, comp_sub, stats);
                        if(dist<best_dist) {
                            best_dist = dist;
                        }
                    }
                    L_INFO << "Best dist: " << best_dist;
                    return;
                }

            }
        }

    }
}


}
}