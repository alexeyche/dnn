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

Orderline::SplitStat Orderline::makeBestSplit(const Dataset &d) const {
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

        double gain = calcInformationGain(
            left_classes_hist
          , right_classes_hist
          , N_left
          , N_right
          , d.entropy()
          , d.N()
        );

        double gap = sumDistRight/N_right - sumDistLeft/N_left;

        if(betterSplitStats(split_stat.gain, split_stat.gap, gain, gap)) {
            split_stat.gain = gain;
            split_stat.gap = gap;
            split_stat.split_position = split_pos;
            split_stat.left_classes_hist = left_classes_hist;
            split_stat.right_classes_hist = right_classes_hist;
            split_stat.N_left = N_left;
            split_stat.N_right = N_right;
        }
    }
    split_stat.split_dist = splitDist(split_stat.split_position);

    for(size_t ci=0; ci < d.C(); ++ci) {
        // L_INFO << (double) split_stat.left_classes_hist[ci] / (double) split_stat.N_left << " < " << (double) split_stat.right_classes_hist[ci] / (double) split_stat.N_right;
        if( ( (double) split_stat.left_classes_hist[ci] / (double) split_stat.N_left ) >
            ( (double) split_stat.right_classes_hist[ci] / (double) split_stat.N_right ) ) {
        // if(split_stat.left_classes_hist[ci]>split_stat.right_classes_hist[ci]) {
            split_stat.left_major_classes.insert(ci);
            // L_INFO << "left major!";
        } else
        if(split_stat.right_classes_hist[ci] > 0) {
            split_stat.right_major_classes.insert(ci);
            // L_INFO << "right major!";
        }
    }

    return split_stat;
}

double Orderline::calcInformationGain(
    const unordered_map<size_t, size_t> &left_hist
  , const unordered_map<size_t, size_t> &right_hist
  , const size_t &N_left
  , const size_t &N_right
  , const double &dataset_entropy
  , const size_t &N
) {
    double entropy_left = Dataset::calcEntropy(left_hist, N_left);
    double entropy_right = Dataset::calcEntropy(right_hist, N_right);
    double ent = dataset_entropy - N_left * entropy_left / N - N_right * entropy_right / N;

    // L_DEBUG << ent << " == " << dataset_entropy << " - " << N_left << "*" << entropy_left << "/" << N << " - " << N_right << "*" << entropy_right << "/" << N;
    return ent;
}


const double Orderline::splitDist(const size_t &split_pos) const {
    if(split_pos+1 == projs.size()) return 0.0;
    return (projs[split_pos+1].dist + projs[split_pos].dist)/2.0;
}

Orderline& Orderline::operator=(const Orderline& other) {
    _all_min = other._all_min;
    projs = other.projs;
    return *this;
}

double Orderline::maxPossibleFurtherInformationGain(const Dataset &d) const {
    SplitStat split_stat = makeBestSplit(d);

    L_INFO << split_stat;
    L_INFO << "left_hist before: ";
    printHist(split_stat.left_classes_hist, L_INFO);
    //for(const auto &v: split_stat.left_classes_hist) L_INFO << v.first << ": " << v.second;
    L_INFO << "right_hist before: ";
    printHist(split_stat.right_classes_hist, L_INFO);

    unordered_map<size_t, size_t> new_left_hist(split_stat.left_classes_hist);
    unordered_map<size_t, size_t> new_right_hist(split_stat.right_classes_hist);
    const unordered_map<size_t, size_t>& class_hist = d.class_hist();

    for(size_t ci=0; ci<d.C(); ++ci) {
        size_t remained = class_hist.at(ci) - split_stat.left_classes_hist[ci] - split_stat.right_classes_hist[ci];
        if(split_stat.leftMajorHas(ci)) {
            new_left_hist[ci] += remained;
        } else
        if(split_stat.rightMajorHas(ci)) {
            new_right_hist[ci] += remained;
        }
    }
    L_INFO << "left_hist after: ";
    printHist(new_left_hist, L_INFO);
    L_INFO << "right_hist after: ";
    printHist(new_right_hist, L_INFO);


    size_t N_left = 0;
    size_t N_right = 0;

    for(const auto &cell: new_left_hist) {
        N_left += cell.second;
    }
    for(const auto &cell: new_right_hist) {
        N_right += cell.second;
    }

    double max_gain = calcInformationGain(
        new_left_hist
      , new_right_hist
      , N_left
      , N_right
      , d.entropy()
      , d.N()
    );
    return max_gain;
}

double Orderline::maxPossibleInformationGain(const SplitStat &split_stat, const double &range, const Dataset &d) const {
    // L_INFO << split_stat;
    // L_INFO << "left_hist before: ";
    // for(const auto &v: split_stat.left_classes_hist) L_INFO << v.first << ": " << v.second;
    // L_INFO << "right_hist before: ";
    // for(const auto &v: split_stat.right_classes_hist) L_INFO << v.first << ": " << v.second;

    double max_gain = 0.0;
    for(size_t pi = 0; pi < projs.size()-1; ++pi) {
        double tau = (projs[pi].dist + projs[pi+1].dist)/2.0;
        // L_INFO << "tau: " << tau;
        unordered_map<size_t, size_t> new_left_hist(split_stat.left_classes_hist);
        unordered_map<size_t, size_t> new_right_hist(split_stat.right_classes_hist);

        for(const auto &p: projs) {
            if(p.dist>tau+range) break; // because it's sorted
            if(p.dist<tau-range) continue;

            // L_INFO << p.class_id;
            // L_INFO << "left_hist before c: ";
            // for(const auto &v: new_left_hist) L_INFO << v.first << ": " << v.second;
            // L_INFO << "right_hist before c: ";
            // for(const auto &v: new_right_hist) L_INFO << v.first << ": " << v.second;
            // L_INFO << "(" << p.dist << " > " << tau << ") && " << split_stat.leftMajorHas(p.class_id);

            if( (p.dist > tau) && split_stat.leftMajorHas(p.class_id) && new_right_hist[p.class_id] > 0) {
                // L_INFO << "Moving " << p.class_id << " to the left major";

                new_left_hist[p.class_id]++;
                new_right_hist[p.class_id]--;
            } else
            if(split_stat.rightMajorHas(p.class_id) && new_left_hist[p.class_id] > 0) {
                // L_INFO << "Moving " << p.class_id << " to the right major";
                new_right_hist[p.class_id]++;
                new_left_hist[p.class_id]--;
            }

            // L_INFO << "left_hist after c: ";
            // for(const auto &v: new_left_hist) L_INFO << v.first << ": " << v.second;
            // L_INFO << "right_hist after c: ";
            // for(const auto &v: new_right_hist) L_INFO << v.first << ": " << v.second;

        }

        // L_INFO << "left_hist after: ";
        // for(const auto &v: new_left_hist) L_INFO << v.first << ": " << v.second;
        // L_INFO << "right_hist after: ";
        // for(const auto &v: new_right_hist) L_INFO << v.first << ": " << v.second;

        size_t N_left = 0;
        size_t N_right = 0;

        for(const auto &cell: new_left_hist) {
            N_left += cell.second;
        }
        for(const auto &cell: new_right_hist) {
            N_right += cell.second;
        }

        double new_gain = calcInformationGain(
            new_left_hist
          , new_right_hist
          , N_left
          , N_right
          , d.entropy()
          , d.N()
        );
        max_gain = fmax(max_gain, new_gain);
    }

    if(max_gain < split_stat.gain) {
        throw dnnException() << "You algorithm sucks\n";
    }

    return max_gain;
}




}
}