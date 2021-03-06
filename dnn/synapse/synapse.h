#pragma once

#include <ground/serial/proto_serial.h>
#include <dnn/protos/synapse.pb.h>
#include <dnn/protos/config.pb.h>
#include <ground/rand.h>

namespace NDnn {
	using namespace NGround;
	
	struct TSynapseInnerState: public IProtoSerial<NDnnProto::TSynapseInnerState> {
		void SerialProcess(TProtoSerial& serial) override {
			serial(IdPre);
			serial(DendriteDelay);
			serial(Weight);
			serial(Potential);
			serial(Fired);
			serial(PostSynapticWeight);
			serial(LearningRate);
		}

		size_t IdPre = 0;
		double DendriteDelay = 0.0;
		double Weight = 1.0;
		double Potential = 0.0;
		bool Fired = false;
		double PostSynapticWeight = 1.0;
		double LearningRate = 1.0;
	};

	template <typename TConstants, typename TState>
	class TSynapse: public IProtoSerial<NDnnProto::TLayer> {
	public:
		using TConst = TConstants;
		TSynapse()
		{}

		bool& MutFired() {
			return InnerState.Fired;
		}

		const bool& Fired() const {
			return InnerState.Fired;
		}

		double& MutPostSynapticWeight() {
			return InnerState.PostSynapticWeight;
		}
		const double& PostSynapticWeight() const {
			return InnerState.PostSynapticWeight;
		}


		double& MutPotential() {
			return InnerState.Potential;
		}
		const double& Potential() const {
			return InnerState.Potential;
		}

		double WeightedPotential() const {
			return InnerState.Weight * InnerState.Potential * InnerState.PostSynapticWeight;
		}

		void SerialProcess(TProtoSerial& serial) override final {
			serial(c, TConstants::ProtoFieldNumber);
			serial(s, TState::ProtoFieldNumber);
			serial(InnerState, NDnnProto::TLayer::kSynapseInnerStateFieldNumber);
		}

		double& MutWeight() {
			return InnerState.Weight;
		}
		
		const double& Weight() const {
			return InnerState.Weight;
		}

		ui32& MutIdPre() {
			return InnerState.IdPre;
		}

		const ui32& IdPre() const {
			return InnerState.IdPre;
		}

		double& MutDendriteDelay() {
			return InnerState.DendriteDelay;
		}

		const double& DendriteDelay() const {
			return InnerState.DendriteDelay;
		}

		TConstants& MutConstants() {
			return c;
		}

		const TState& State() const {
			return s;
		}

		void SetRandEngine(TRandEngine& rand) {
			Rand.Set(rand);
		}

		const double& LearningRate() const {
			return InnerState.LearningRate;
		}

		double& MutLearningRate() {
			return InnerState.LearningRate;
		}
	private:
		TSynapseInnerState InnerState;

	protected:
		TPtr<TRandEngine> Rand;

		TConstants c;
		TState s;
	};


} // namespace NDnn


