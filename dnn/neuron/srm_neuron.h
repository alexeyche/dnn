#pragma once

#include "spike_neuron.h"

#include <ground/serial/proto_serial.h>
#include <dnn/protos/srm_neuron.pb.h>
#include <dnn/protos/config.pb.h>

namespace NDnn {

	struct TSRMNeuronConst: public IProtoSerial<NDnnProto::TSRMNeuronConst> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kSRMNeuronFieldNumber;

        void SerialProcess(TProtoSerial& serial) override final {
			serial(Urest);
		    serial(AmpRefr);
		    serial(AmpAdapt);
		    serial(TauRefr);
		    serial(TauAdapt);
        }


	    double Urest = 0.0;
	    double AmpRefr = 100.0;
	    double AmpAdapt = 1.0;
	    double TauRefr = 2.0;
	    double TauAdapt = 50.0;
	};


	struct TSRMNeuronState: public IProtoSerial<NDnnProto::TSRMNeuronState> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kSRMNeuronStateFieldNumber;

	    void SerialProcess(TProtoSerial& serial) override final {
            serial(VarRefr);
            serial(VarAdapt);
        }

	    double VarRefr = 0.0;
	    double VarAdapt = 0.0;
	};

	class TSRMNeuron: public TSpikeNeuron<TSRMNeuronConst, TSRMNeuronState> {
	public:
		void Reset() {
	        MutMembrane() = c.Urest;
	        MutSpikeProbability() = 0.0;
	        s.VarRefr = 0.0;
	    }

	    void PostSpikeDynamics(const TTime& t) {
	        s.VarRefr += c.AmpRefr;
	        s.VarAdapt += c.AmpAdapt;
	    }

	    void CalculateDynamics(const TTime& t, double Iinput, double Isyn) {
	    	MutMembrane() = c.Urest + Iinput + Isyn;
	        MutProbabilityModulation() = std::exp( - s.VarRefr - s.VarAdapt);

	        s.VarRefr += - t.Dt * s.VarRefr/c.TauRefr;
        	s.VarAdapt += - t.Dt * s.VarAdapt/c.TauAdapt;
	    }
	};

} // namespace NDnn
