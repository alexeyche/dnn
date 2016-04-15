#pragma once

#include "reinforcement.h"

#include <dnn/protos/input_classifier.pb.h>
#include <dnn/protos/config.pb.h>
#include <dnn/sim/global_ctx.h>

namespace NDnn {

    struct TInputClassifierConst: public IProtoSerial<NDnnProto::TInputClassifierConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kInputClassifierFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(Ltp);
            serial(Ltd);
        }

        double Ltp = 1.0;
        double Ltd = -1.0;
    };

    template <typename TNeuron>
    class TInputClassifier: public TReinforcement<TInputClassifierConst, TNeuron> {
    public:
        using TPar = TReinforcement<TInputClassifierConst, TNeuron>;

        void ModulateReward() {
            if (TPar::Neuron().Fired()) {
                TMaybe<ui32> currentClassId = TGlobalCtx::Inst().GetCurrentClassId();
                
                if (currentClassId) {
                    if (currentClassId.GetRef() == TPar::SpaceInfo().LocalId) {
                        TGlobalCtx::Inst().PropagateReward(TPar::c.Ltp);
                    } else {
                        TGlobalCtx::Inst().PropagateReward(TPar::c.Ltd);
                    }
                }
            }
        }
    };

} // namespace NDnn
