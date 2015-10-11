#pragma once


#include "reinforcement.h"

#include <dnn/protos/input_classifier.pb.h>

namespace dnn {

/*@GENERATE_PROTO@*/
struct InputClassifierC : public Serializable<Protos::InputClassifierC> {
    InputClassifierC()
    : ltp(1.0)
    , ltd(-1.0)
    {
    }

    void serial_process() {
        begin() << "ltp: " << ltp << ", "
                << "ltd: " << ltd << Self::end;
    }

    double ltp;
    double ltd;
};


class InputClassifier : public Reinforcement<InputClassifierC> {
public:
    const string name() const {
        return "InputClassifier";
    }
    void modulateReward() {
        if(n->fired()) {
            if(n->localId() == GlobalCtx::inst().getCurrentClassId()) {
                GlobalCtx::inst().propagateReward(c.ltp);
            } else {
                GlobalCtx::inst().propagateReward(c.ltd);
            }
        }
    }
};



}