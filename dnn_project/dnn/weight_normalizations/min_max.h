#pragma once


#include "weight_normalization.h"

#include <dnn/protos/min_max.pb.h>

namespace dnn {

/*@GENERATE_PROTO@*/
struct MinMaxC : public Serializable<Protos::MinMaxC> {
    MinMaxC()
    : min_weight(0.0)
    , max_weight(1.0)
    {
    }

    void serial_process() {
        begin() << "min_weight: " << min_weight << ", "
                << "max_weight: " << max_weight << Self::end;
    }

    double min_weight;
    double max_weight;
};


class MinMax : public WeightNormalization<MinMaxC> {
public:
    double derivativeModulation(const double &w) {
	    if((w >= c.max_weight)||(w <= c.min_weight)) {
	    	return 0.0;
	    }
        return 1.0;
    }
};



}
