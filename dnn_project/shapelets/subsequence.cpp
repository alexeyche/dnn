#include "subsequence.h"

#include <dnn/util/time_series.h>

namespace dnn {
namespace shapelets {



Ptr<TimeSeries> Subsequence::cutTimeSeries() {
    assert(_referent.isSet() && _length > 0);

    Ptr<TimeSeries> cut(Factory::inst().createObject<TimeSeries>());
    for(size_t di=0; di<_referent->dim(); ++di) {
        const vector<double> v = _referent->getVector(di);
        for(size_t i=_from; i<(_from+_length); ++i) {
            cut->addValue(di, v[i]);
        }
    }
    return cut;
}

void Subsequence::serial_process() {
    Ptr<TimeSeries> cut;
    if(mode == ProcessingOutput) {
        cut = cutTimeSeries();
    } else {
        cut = Factory::inst().createObject<TimeSeries>();
        _referent.set(cut.ptr());
    }
    begin() << "Values: " << cut.ref() << Self::end;
}

size_t Subsequence::dim() const {
    return _referent->dim();
}

const double& Subsequence::operator () (const size_t &dim, const size_t &id) const {
    return _referent.ref()(dim, id);
}

}
}