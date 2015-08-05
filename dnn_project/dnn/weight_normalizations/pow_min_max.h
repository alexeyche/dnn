#pragma once


#include "weight_normalization.h"

#include <dnn/protos/pow_min_max.pb.h>
#include <dnn/util/fastapprox/fastpow.h>

namespace dnn {

/*@GENERATE_PROTO@*/
struct PowMinMaxC : public Serializable<Protos::PowMinMaxC> {
    PowMinMaxC() 
    : power(4.0)
    , mean_weight(0.2)
    , min_weight(0.0)
    , max_weight(1.0)
    {
    }

    void serial_process() {
        begin() << "power: " << power << ", " 
                << "mean_weight: " << mean_weight << ", "
                << "min_weight: " << min_weight << ", "
                << "max_weight: " << max_weight << Self::end;

    	__mean_weight = fastpow(mean_weight, power);            
    }

    double power;
    double mean_weight;
    double __mean_weight;
    double min_weight;
    double max_weight;
};


class PowMinMax : public WeightNormalization<PowMinMaxC> {
public:
    const string name() const {
        return "PowMinMax";
    }
    double derivativeModulation(const double &w) {
	    if((w >= c.max_weight)||(w <= c.min_weight)) {
	    	return 0.0;
	    }
        double wp = fastpow(w, c.power);
        return (wp/(wp+c.__mean_weight));
    }
};



}
