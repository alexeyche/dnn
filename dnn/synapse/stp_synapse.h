#pragma once

#include "synapse.h"

#include <dnn/util/serial/proto_serial.h>
#include <dnn/protos/stp_synapse.pb.h>
#include <dnn/protos/config.pb.h>


namespace NDnn {

    struct TSTPSynapseConst: public IProtoSerial<NDnnProto::TSTPSynapseConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kSTPSynapseFieldNumber;

        void SerialProcess(TProtoSerial& serial) override {
            serial(D);
            serial(F);
            serial(MaxWeight);
        }

        double D = 200.0;
        double F = 50.0;
        double MaxWeight = 1.0;
    };

    struct TSTPSynapseState: public IProtoSerial<NDnnProto::TSTPSynapseState> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kSTPSynapseStateFieldNumber;

        void SerialProcess(TProtoSerial& serial) override {
            serial(r);
            serial(p);
        }

        double r = 1.0;
        double p = 0.0;
    };

    class TSTPSynapse : public TSynapse<TSTPSynapseConst, TSTPSynapseState> {
    public:
        void Reset() {
            s.r = 1.0;
            s.p = 0.0;
        }

        void CalculateDynamics(const TTime &t) {
            s.r += t.Dt * (1 - s.r)/c.D;
            s.p += t.Dt * (Weight() - s.p)/c.F;
        }

    	void PropagateSpike() {
    	    s.r += - std::abs(s.p) * s.r;
            s.p += Weight() * (c.MaxWeight - std::abs(s.p));
    	}

        double Potential() const {
            return s.p * s.r;
        }

        double WeightedPotential() const {
            return Weight() * Potential();
        }

    };

} // namespace NDnn
