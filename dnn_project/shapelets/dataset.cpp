
#include "dataset.h"

#include <dnn/util/time_series.h>


namespace dnn {
namespace shapelets {

Dataset::Dataset(vector<Ptr<TimeSeries>> &_ts)
 : ts_set(_ts)
{
    size_t class_id = 0;
    for(auto &v: ts_set) {
        for(const auto &l_id: v->info.labels_ids) {
            const string &lab = v->info.unique_labels[l_id];
            {
                auto p = class_ids.find(lab);
                if(p == class_ids.end()) {
                    class_ids.insert(std::make_pair(lab, class_id++));
                }
            }
            pushIntoHist(class_counts, class_ids[lab]);

        }
    }
    if(class_counts.size() <= 1) {
        throw dnnException() << "Got one or zero classes in dataset. Running shapelet pointeless\n";
    }
    L_INFO << "Created Dataset, with class counts:";
    for(auto &v: class_counts) {
        L_INFO << "\t" << v.first << ": " << v.second;
    }
    _entropy = calcEntropy(class_counts, N());

    L_INFO << "Got dataset entropy: " << entropy();
}

double Dataset::calcEntropy(const unordered_map<size_t, size_t> &_class_counts, const size_t &N) {
    double E = 0.0;
    for(const auto &cc: _class_counts) {
        const size_t &counts = cc.second;
        if(counts == 0) {
            continue;
        }
        double ratio = (double)counts/N;
        E -= ratio * log(ratio);
    }
    return E;
}

Dataset::~Dataset() {
}

const size_t& Dataset::getTsLabelId(const size_t& id) {
    return class_ids[(*this)(id)->getLabel()];
}



}
}