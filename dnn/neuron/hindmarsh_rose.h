#pragma once

#include <cmath>

#include "spike_neuron.h"

#include <ground/serial/proto_serial.h>
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
            serial(Phi1);
            serial(Phi2);
            serial(Gy);
            serial(Gz);
            serial(Alpha);
            serial(Phi3);
            serial(Beta);
        }

        double A = 1.0;
        double B = 0.0;
        double C = 0.8;
        double D = 1.0;
        double R = 0.005;
        double S = 4.0;
        double XR = -2.618;
        double Phi1 = 3.0;
        double Phi2 = 2.0;
        double Gy = 5.0;
        double Gz = 1.0;
        double Alpha = 1.0;
        double Phi3 = 2.0;
        double Beta = 1.0;
	};


	struct THindmarshRoseState: public IProtoSerial<NDnnProto::THindmarshRoseState> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kHindmarshRoseStateFieldNumber;
		void SerialProcess(TProtoSerial& serial) override final {
            serial(SpikingVariable);
            serial(BurstingVariable);
            serial(I);
        }

        double SpikingVariable = 0.0;
        double BurstingVariable = 0.0;
        double I = 0.0;
	};

	class THindmarshRose : public TSpikeNeuron<THindmarshRoseConst, THindmarshRoseState> {
	public:
		void Reset() {
	    }

	    void PostSpikeDynamics(const TTime& t) {
	    }

	    void CalculateDynamics(const TTime& t, double Iinput, double Isyn) {
	        MutMembrane() += t.Dt * (- c.A * pow(Membrane(), 3) + c.B * pow(Membrane(), 2) + c.Phi1 * Membrane() + c.Phi2 + c.Gy * s.SpikingVariable - c.Gz * s.BurstingVariable + c.Alpha * (Iinput + Isyn));
	        s.SpikingVariable += t.Dt * ( -c.C - c.D * pow(Membrane(), 2) - c.Phi3 * Membrane() - c.Beta * s.SpikingVariable);
	        s.BurstingVariable += t.Dt * c.R *( c.S * ( Membrane() - c.XR) - s.BurstingVariable);
            s.I = (Iinput + Isyn);
	    }
	};

} // namespace NDnn
