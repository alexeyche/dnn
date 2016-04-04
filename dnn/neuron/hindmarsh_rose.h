#pragma once

#include <cmath>

#include "spike_neuron.h"

#include <dnn/util/serial/proto_serial.h>
#include <dnn/protos/hindmarsh_rose.pb.h>
#include <dnn/protos/config.pb.h>

namespace NDnn {

	struct THindmarshRoseConst: public IProtoSerial<NDnnProto::THidmarshRoseConst> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kHindmarshRoseFieldNumber;

        void SerialProcess(TProtoSerial& serial) override final {
            serial(A);
            serial(B);
            serial(C);
            serial(D);
            serial(R);
            serial(S);
            serial(XR);
            serial(SpikingVariable);
            serial(BurstingVariable);
        }

        double A = 1.0;
        double B = 3.0;
        double C = 1.0;
        double D = 5.0;
        double R = 0.001;
        double S = 4;
        double XR = - 1.6;
        double SpikingVariable = 0.0;
        double BurstingVariable = 0.0;
	};


	struct TIntegrateAndFireState: public IProtoSerial<NDnnProto::TIntegrateAndFireState> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kIntegrateAndFireStateFieldNumber;
	};

	class TIntegrateAndFire : public TSpikeNeuron<TIntegrateAndFireConst, TIntegrateAndFireState> {
	public:
		void Reset() {
	    }

	    void PostSpikeDynamics(const TTime& t) {
	    }

		const double& SpikingVariable() const {
			return c.SpikingVariable;
		}

		const double& BurstingVariable() const {
			return c.BurstingVariable;
		}

		double& MutSpikingVariable() {
			return c.SpikingVariable;
		}

		double& MutBurstingVariable() {
			return c.BurstingVariable;
		}

	    void CalculateDynamics(const TTime& t, double Iinput, double Isyn) {
	        MutMembrane() += t.Dt * ( SpikingVariable()  - c.A * pow(Membrane(), 3) + c.B * pow(Membrane(), 2) - BurstingVariable() + Iinput + Isyn);
	        MutSpikingVaribale() += t.Dt * ( c.C - c.D * pow(Membrane(), 2)  - SpikingVariable());
	        MutBurstingVariable() += t.Dt * c.R *( c.S * ( Membrane() - c.XR) - BurstingVariable());
	    }
	};

} // namespace NDnn
