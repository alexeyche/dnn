#pragma once

#include "reinforcement.h"

#include <dnn/protos/sequence_likelihood.pb.h>
#include <dnn/protos/config.pb.h>
#include <dnn/sim/global_ctx.h>

namespace NDnn {

    struct TSequenceLikelihoodConst: public IProtoSerial<NDnnProto::TSequenceLikelihoodConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kSequenceLikelihoodFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
        }
    };

    namespace {

        double SafeLog(double v) {
            if (v <= 0.0) {
                return 0.0;
            }
            return std::log(v);
        }

    }

    template <typename TNeuron>
    class TSequenceLikelihood: public TReinforcement<TSequenceLikelihoodConst, TNeuron> {
    public:
        using TPar = TReinforcement<TSequenceLikelihoodConst, TNeuron>;

        void ModulateReward(const TTime& t) {
            if (GetTarget(t)) {
                TGlobalCtx::Inst().PropagateReward(SafeLog(TPar::Neuron().SpikeProbability()));
                ++CurrentId;
            } else {
                TGlobalCtx::Inst().PropagateReward(SafeLog(1.0 - TPar::Neuron().SpikeProbability()));
            }
        }

        bool GetTarget(const TTime& t) const {
            ENSURE(TargetSet, "Need target be set");
            bool val = false;
            // if (CurrentId < Target.size()) L_INFO << CurrentId << " " << Target.size() << " " << t.T << " >= " << Target[CurrentId] << ", " << t.T+t.Dt << " < " << Target[CurrentId];
            while ((CurrentId < Target.size()) && (std::numeric_limits<double>::epsilon() + Target[CurrentId] >= t.T) && (Target[CurrentId] < (t.T+t.Dt))) {
                val = true;
            }
            // L_INFO << t.T << " " << val;
            return val;
        }

        void SetTarget(const TVector<double>& targetSeq) {
            Target = targetSeq;
            TargetSet = true;
            CurrentId = 0;
        }

        double GetModulation() {
            if (TargetSet) {
                return 1.0;
            }
        }

    private:
        bool TargetSet = false;
        TVector<double> Target;
        mutable ui32 CurrentId = 0;
    };

} // namespace NDnn
