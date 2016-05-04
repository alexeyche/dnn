#pragma once

#include <cmath>

#include "spike_neuron.h"

#include <dnn/util/serial/proto_serial.h>
#include <dnn/protos/hindmarsh_rose.pb.h>
#include <dnn/protos/config.pb.h>

namespace NDnn {

	struct THindmarshRoseConst: public IProtoSerial<NDnnProto::THindmarshRoseConst> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kHindmarshRoseFieldNumber;

        void SerialProcess(TProtoSerial& serial) override final {
            serial(A);
            serial(B);
            serial(C);
            serial(D);
            serial(R);
            serial(S);
            serial(XR);
        }

        double A = 1.0;
        double B = 3.0;
        double C = 1.0;
        double D = 5.0;
        double R = 0.001;
        double S = 4;
        double XR = -1.6;
	};


	struct THindmarshRoseState: public IProtoSerial<NDnnProto::THindmarshRoseState> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kHindmarshRoseStateFieldNumber;
		void SerialProcess(TProtoSerial& serial) override final {
            serial(SpikingVariable);
            serial(BurstingVariable);
        }

        double SpikingVariable = 0.0;
        double BurstingVariable = 0.0;
	};

	class THindmarshRose : public TSpikeNeuron<THindmarshRoseConst, THindmarshRoseState> {
	public:
		void Reset() {
	    }

	    void PostSpikeDynamics(const TTime& t) {
	    }

	    void CalculateDynamics(const TTime& t, double Iinput, double Isyn) {
	        MutMembrane() += t.Dt * ( s.SpikingVariable  - c.A * pow(Membrane(), 3) + c.B * pow(Membrane(), 2) - s.BurstingVariable + Iinput + Isyn);
	        s.SpikingVariable += t.Dt * ( c.C - c.D * pow(Membrane(), 2)  - s.SpikingVariable);
	        s.BurstingVariable += t.Dt * c.R *( c.S * ( Membrane() - c.XR) - s.BurstingVariable);
	    }
	};

} // namespace NDnn
