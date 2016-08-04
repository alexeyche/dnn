#pragma once

#include "synapse.h"
#include "dnn/sim/global_ctx.h"

#include <ground/serial/proto_serial.h>
#include <dnn/protos/hedonistic_synapse.pb.h>
#include <dnn/protos/config.pb.h>


namespace NDnn {

    struct THedonisticSynapseConst: public IProtoSerial<NDnnProto::THedonisticSynapseConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kHedonisticSynapseFieldNumber;

        void SerialProcess(TProtoSerial& serial) override {
            serial(PspDecay);
            serial(Amp);
            serial(TauRef);
            serial(DeltaCatalyst);
            serial(TauCatalyst);
            serial(TauEligibility);
            serial(LearningRate);
        }

        double PspDecay = 10.0;
        double Amp = 1.0;
        double TauRef = 1.0;
        double DeltaCatalyst = 0.0;
        double TauCatalyst = 1.0;
        double TauEligibility = 15.0;
        double LearningRate = 0.1;
    };

    struct THedonisticSynapseState: public IProtoSerial<NDnnProto::THedonisticSynapseState> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kHedonisticSynapseStateFieldNumber;

        void SerialProcess(TProtoSerial& serial) override {
            serial(r);
            serial(p);
            serial(c);
            serial(q);
            serial(e);
        }

        double r = 0.0;
        double p = 0.5;
        double c = 0.0;     // c, catalyst
        double q = 0.0;     // q, probability weight
        double e = 0.0;     // e, eligibility trace
    };

    class THedonisticSynapse : public TSynapse<THedonisticSynapseConst, THedonisticSynapseState> {
    public:
        void Reset() {
            MutPotential() = 0;
            s.r = 0;
        }

        void CalculateDynamics(const TTime &t) {
            MutPotential() += - t.Dt * Potential()/c.PspDecay;

            if (!(s.r < 0.1))
                s.r += - t.Dt;
            s.e += - t.Dt * s.e/c.TauEligibility;
            s.c += - t.Dt * s.c/c.TauCatalyst;
            s.q += c.LearningRate * s.e * TGlobalCtx::Inst().GetRewardDelta();

        }

        void PropagateSpike() {
            if (s.r < 0.1) {
                s.p = 1.0 / (1.0 + exp(-s.q - s.c));
                if (s.p > Rand->GetUnif()) {
                    // release
                    MutPotential() += c.Amp;
                    s.e += 1 - s.p;
                    s.r = c.TauRef;
                    s.c += c.DeltaCatalyst;
                }
                else {
                    // failure
                    s.e += -s.p;
                }
            }
        }
    };
    
} // namespace NDnn