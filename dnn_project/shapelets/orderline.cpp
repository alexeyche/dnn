#include "orderline.h"

#include "dataset.h"
#include "dist.h"

namespace dnn {
namespace shapelets {

void Orderline::insert(Projection &&newp) {
    if(fabs(newp.dist - DistAlgorithm::MIN_DIST) > 1e-10) {
        _all_min = false;
    }
    for(auto ptr = projs.begin(); ptr != projs.end(); ++ptr) {
        if(newp.dist <= ptr->dist) {
            projs.insert(ptr, newp);
            return;
        }
    }
    projs.push_back(newp);
}

Orderline::SplitStat Orderline::makeBestSplit() const {
    if(illConditioned()) {
        throw dnnException() << "Trying to do best split on ill conditioned orderline\n";
    }
    SplitStat split_stat;
    for(size_t split_pos = 0; split_pos < projs.size(); ++split_pos) {
        unordered_map<size_t, size_t> left_classes_hist(d.C());
        unordered_map<size_t, size_t> right_classes_hist(d.C());

        size_t N_left = split_pos + 1;
        size_t N_right = projs.size() - N_left;

        double sumDistLeft = 0.0, sumDistRight = 0.0;
        for(size_t pi=0; pi<N_left; ++pi) {
            const Projection &left_prj = projs[pi];

            Dataset::pushIntoHist(left_classes_hist, left_prj.class_id);
            sumDistLeft += left_prj.dist;
        }
        for(size_t pi=N_left; pi<projs.size(); ++pi) {
            const Projection &right_prj = projs[pi];

            Dataset::pushIntoHist(right_classes_hist, right_prj.class_id);
            sumDistRight += right_prj.dist;
        }

        // for(size_t ci=0; ci < d.C(); ++ci) {
        //     if( ( (double)left_classes_hist[ci] / (double) N_left ) >
        //         ( (double)right_classes_hist[ci] / (double) N_right ) ) {
        //         split_stat.left_majority.push_back(ci);
        //     } else {
        //         split_stat.right_majority.push_back(ci);
        //     }
        // }
        double entropy_left = Dataset::calcEntropy(left_classes_hist, d.N());
        double entropy_right = Dataset::calcEntropy(right_classes_hist, d.N());

        // L_DEBUG << d.entropy() << " - " << N_left << "*" << entropy_left << "/" << d.N() << " - " << N_right << "*" << entropy_right << "/" << d.N();

        double gain = d.entropy() - N_left * entropy_left / d.N() - N_right * entropy_right / d.N();
        double gap = sumDistRight/N_right - sumDistLeft/N_left;

        if(betterSplitStats(split_stat.gain, split_stat.gap, gain, gap)) {
            split_stat.gain = gain;
            split_stat.gap = gap;
            split_stat.split_position = split_pos;
        }
    }
    split_stat.split_dist = splitDist(split_stat.split_position);

    return split_stat;
}

const double Orderline::splitDist(const size_t &split_pos) const {
    if(split_pos+1 == projs.size()) return 0.0;
    return (projs[split_pos+1].dist + projs[split_pos].dist)/2.0;
}


}
}