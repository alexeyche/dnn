#pragma once

#include <unordered_map>
#include <vector>

using std::vector;
using std::unordered_map;

#include <dnn/util/ptr.h>
#include <dnn/util/log/log.h>

namespace dnn {
class TimeSeries;

namespace shapelets {

class Dataset {
public:
    Dataset(vector<Ptr<TimeSeries>> &_ts);
    ~Dataset();
    size_t N() const {
        return ts_set.size();
    }
    size_t C() const {
        return class_counts.size();
    }

    Ptr<TimeSeries> operator() (const size_t &i) const {
        return ts_set[i];
    }
    vector<Ptr<TimeSeries>>::iterator begin() {
        return ts_set.begin();
    }
    vector<Ptr<TimeSeries>>::iterator end() {
        return ts_set.end();
    }
    vector<Ptr<TimeSeries>>::const_iterator begin() const {
        return ts_set.cbegin();
    }
    vector<Ptr<TimeSeries>>::const_iterator end() const {
        return ts_set.cend();
    }
private:
    unordered_map<string, size_t> class_counts;
    unordered_map<string, size_t> class_ids;
    vector<Ptr<TimeSeries>> ts_set;
};



}
}
