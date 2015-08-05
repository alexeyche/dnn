
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
                auto p = class_counts.find(lab);
                if(p == class_counts.end()) {
                    class_counts.insert(make_pair(lab, 1));
                } else {
                    ++(p->second);
                }
            }
            {
                auto p = class_ids.find(lab);
                if(p == class_ids.end()) {
                    class_ids.insert(make_pair(lab, class_id++));
                }
            }
        }
    }
    L_INFO << "Created Dataset, with class counts:";
    for(auto &v: class_counts) {
        L_INFO << "\t" << v.first << ": " << v.second;
    }
}

Dataset::~Dataset() {
}


}
}