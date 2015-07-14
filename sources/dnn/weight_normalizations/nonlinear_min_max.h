#pragma once


#include "weight_normalization.h"

#include <dnn/protos/generated.pb.h>
#include <dnn/util/fastapprox/fastpow.h>

namespace dnn {

/*@GENERATE_PROTO@*/
struct NonLinearMinMaxC : public Serializable<Protos::NonLinearMinMaxC> {
    NonLinearMinMaxC() 
    : depression_factor(1.0)
    , mu(0.5)
    , min_weight(0.0)
    , max_weight(1.0)
    {
    }

    void serial_process() {
        begin() << "depression_factor: " << depression_factor << ", " 
                << "mu: " << mu << ", "
                << "min_weight: " << min_weight << ", "
                << "max_weight: " << max_weight << Self::end;

    }

    double depression_factor;
    double mu;
    double min_weight;
    double max_weight;
};


class NonLinearMinMax : public WeightNormalization<NonLinearMinMaxC> {
public:
    const string name() const {
        return "NonLinearMinMax";
    }
    double ltp(const double &w) {
        return fastpow( 1 - w/c.max_weight, c.mu );
    }
    double ltd(const double &w) {
        return c.depression_factor * fastpow(w/c.max_weight, c.mu);
    }
};



}
