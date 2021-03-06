#pragma once

#include <queue>
#include <atomic>

#include <ground/act_vector.h>
#include <ground/random.h>
#include <ground/rand.h>
#include <ground/optional.h>
#include <ground/serial/meta_proto_serial.h>
#include <ground/serial/util.h>
#include <dnn/protos/config.pb.h>
#include <dnn/protos/spike_neuron_impl.pb.h>


namespace NDnn {
	using namespace NGround;



	struct TNeuronSpaceInfo {
		ui32 LayerId;
		ui32 LocalId;
		ui32 GlobalId;
		ui32 ColumnSize;
		ui32 RowId;
		ui32 ColId;
		ui32 LayerSize;

		bool operator == (const TNeuronSpaceInfo& other) const {
			return GlobalId == other.GlobalId;
		}

		friend std::ostream& operator<<(std::ostream& str, const TNeuronSpaceInfo& self) {
            str << "Neuron(g-" << self.GlobalId << ":layer-" << self.LayerId << ":local-" << self.LocalId << ")";
            return str;
        }
	};

	struct TSpikeNeuronImplInnerState: public IProtoSerial<NDnnProto::TSpikeNeuronImplInnerState> {
		void SerialProcess(TProtoSerial& serial) {
			serial(SynapsesSize);
		}

		ui32 SynapsesSize = 0;
	};

	struct TSpikeNeuronConst: public IProtoSerial<NDnnProto::TSpikeNeuronConst> {
		void SerialProcess(TProtoSerial& serial) {
			serial(AxonDelay);
		}

		double AxonDelay = 0.0;
	};

	class TSpikeNeuronImplInner: public IProtoSerial<NDnnProto::TLayer> {
	public:
		void SerialProcess(TProtoSerial& serial) {
			serial(s, NDnnProto::TLayer::kSpikeNeuronImplInnerStateFieldNumber);
			serial(c, NDnnProto::TLayer::kSpikeNeuronFieldNumber);
		}

		const ui32& SynapsesSize() const {
			return s.SynapsesSize;
		}

		ui32& MutSynapsesSize() {
			return s.SynapsesSize;
		}

		const double& GetAxonDelay() const {
			return c.AxonDelay;
		}

	private:
		TSpikeNeuronImplInnerState s;
		TSpikeNeuronConst c;
	};


	struct TAsyncSpikeQueue {
		TAsyncSpikeQueue() {
			InputSpikesLock.clear();
		}

		TAsyncSpikeQueue(const TAsyncSpikeQueue& other) {
			InputSpikesLock.clear();
			(*this) = other;
		}

		TAsyncSpikeQueue& operator = (const TAsyncSpikeQueue& other) {
			if (this != &other) {
				InputSpikes = other.InputSpikes;
				Info = other.Info;
			}
			return *this;
		}

		void EnqueueSpike(const TSynSpike&& sp) {
			while (InputSpikesLock.test_and_set(std::memory_order_acquire));
			InputSpikes.push(sp);
			InputSpikesLock.clear(std::memory_order_release);
		}

		TNeuronSpaceInfo Info;

		std::priority_queue<TSynSpike> InputSpikes;
		std::atomic_flag InputSpikesLock;
	};


	template <typename TNeuron, typename TConf>
	class TSpikeNeuronImpl: public IMetaProtoSerial {
	public:
		using TNeuronType = TNeuron;
		using TSelf = TSpikeNeuronImpl<TNeuron, TConf>;
		using TConfig = TConf;
		using TReinforcement = typename TConf::template TNeuronReinforcement<TSelf>;

		template <typename TNeuronParameter>
		using TReinforcementParam = typename TConf::template TNeuronReinforcement<TNeuronParameter>;

		void ReadInputSpikes(const TTime &t) {
			while (Queue.InputSpikesLock.test_and_set(std::memory_order_acquire)) {}
			while (!Queue.InputSpikes.empty()) {
		        const TSynSpike& sp = Queue.InputSpikes.top();
		        if(sp.T >= t.T) break;
		        auto& s = Synapses[sp.SynapseId];
		        s.MutFired() = true;
		    	s.PropagateSpike();
		    	LearningRule.PropagateSynapseSpike(sp);

		        Queue.InputSpikes.pop();
		    }
		    Queue.InputSpikesLock.clear(std::memory_order_release);
		}

		void CalculateDynamicsInternal(const TTime& t, double Iinput) {
			ReadInputSpikes(t);

			double Isyn = 0.0;
			auto synIdIt = Synapses.abegin();
		    while (synIdIt != Synapses.aend()) {
		    	auto& synapse = Synapses[synIdIt];
		    	double x = synapse.WeightedPotential();
		    	if (std::fabs(synapse.Potential()) < 0.0001) {
		        	Synapses.SetInactive(synIdIt);
		        } else {
			    	Isyn += x;
		        	++synIdIt;
		        }
		    }
			double Irf = ReceptiveField.CalculateResponse(Iinput);

			Neuron.CalculateDynamics(t, Irf, Isyn);

		    Neuron.MutSpikeProbability() = Activation.SpikeProbability(Neuron.Membrane()) * Neuron.ProbabilityModulation();
			if(t.Dt * Neuron.SpikeProbability() > Rand->GetUnif()) {
		        Neuron.MutFired() = true;
		    }

	    	LearningRule.CalculateDynamics(t);
			LearningRule.MutNorm().CalculateDynamics(t);
			IntrinsicPlasticity.CalculateDynamics(t);
		    Reinforcement.ModulateReward(t);

	   		for(auto syn_id_it = Synapses.abegin(); syn_id_it != Synapses.aend(); ++syn_id_it) {
	        	auto& s = Synapses[syn_id_it];
	        	s.CalculateDynamics(t);
	        	s.MutFired() = false;
	        }
		}

		const TNeuron& GetNeuron() const {
			return Neuron;
		}

		TNeuron& GetNeuron() {
			return Neuron;
		}


		void SetRandEngine(TRandEngine& rand) {
			Rand.Set(rand);
			Neuron.SetRandEngine(rand);

			for (auto& synapse: Synapses) {
				synapse.SetRandEngine(rand);
			}
		}

		void InitReceptiveField(TRandEngine& rand) {
			ReceptiveField.Init(SpaceInfo, rand);
		}

		void Prepare() {
			ENSURE(Rand, "Random engine is not set");

			LearningRule.SetNeuronImpl(*this);
			LearningRule.Reset();
			LearningRule.MutNorm().Reset();

			Reinforcement.SetNeuronImpl(*this);

			IntrinsicPlasticity.SetNeuronImpl(*this);
			IntrinsicPlasticity.Reset();

			Neuron.Reset();
		}

		void SetSpaceInfo(TNeuronSpaceInfo info) {
			SpaceInfo = info;
			Queue.Info = SpaceInfo;
		}

		template <typename TOtherNeuron>
		bool operator == (const TOtherNeuron& other) const {
			return SpaceInfo == other.GetSpaceInfo();
		}

		const TNeuronSpaceInfo& GetSpaceInfo() const {
			return SpaceInfo;
		}

		TAsyncSpikeQueue& GetMutAsyncSpikeQueue() {
			return Queue;
		}

		const ui32& GetGlobalId() const {
			return SpaceInfo.GlobalId;
		}

		const ui32& GetLocalId() const {
			return SpaceInfo.LocalId;
		}

		void AddSynapse(typename TConf::TNeuronSynapse&& syn) {
			Synapses.emplace_back(std::forward<typename TConf::TNeuronSynapse>(syn));
			Inner.MutSynapsesSize()++;
		}

		void SerialProcess(TMetaProtoSerial& serial) override final {
			serial(Neuron);
			serial(Activation);
			serial(Inner);
			serial(ReceptiveField);
			if (serial.IsInput()) {
				Synapses.resize(Inner.SynapsesSize());
				if (Inner.SynapsesSize() == 0) {
					const NDnnProto::TLayer& layerSpec = serial.GetMessage<NDnnProto::TLayer>();
					if (GetRepeatedFieldSizeFromMessage(layerSpec, TConf::TNeuronSynapse::TConst::ProtoFieldNumber) > GetLocalId()) {
						PredefineSynapseConst.emplace(
							GetRepeatedFieldFromMessage<typename TConf::TNeuronSynapse::TConst::TProto>(
								layerSpec,
								TConf::TNeuronSynapse::TConst::ProtoFieldNumber,
								GetLocalId()
							)
						);
					}
				}
			}
			for (ui32 synId = 0; synId < Inner.SynapsesSize(); ++synId) {
				serial(Synapses[synId]);
			}
			serial(LearningRule);
			serial(LearningRule.MutNorm());
			serial(IntrinsicPlasticity);
			serial(Reinforcement);
		}

		const auto& GetPredefinedSynapseConst() const {
			return PredefineSynapseConst;
		}

		const auto& GetSynapses() const {
			return Synapses;
		}

		auto& GetMutSynapses() {
			return Synapses;
		}

		const double& GetAxonDelay() const {
			return Inner.GetAxonDelay();
		}

		const auto& GetLearningRule() const {
			return LearningRule;
		}

		auto& GetMutLearningRule() {
			return LearningRule;
		}

		auto& GetMutActivationFunction() {
			return Activation;
		}

		const auto& GetActivationFunction() const {
			return Activation;
		}

		auto& GetMutWeightNormalization() {
			return LearningRule.MutNorm();
		}

		const auto& GetReinforcement() const {
			return Reinforcement;
		}

		auto& GetMutReinforcement() {
			return Reinforcement;
		}
	private:
		TPtr<TRandEngine> Rand;

		TAsyncSpikeQueue Queue;

		TNeuron Neuron;

		typename TConf::TNeuronActivationFunction Activation;
		typename TConf::TNeuronReceptiveField ReceptiveField;
		typename TConf::template TNeuronLearningRule<TSelf> LearningRule;
		typename TConf::template TNeuronIntrinsicPlasticity<TSelf> IntrinsicPlasticity;
		TReinforcement Reinforcement;

		TActVector<typename TConf::TNeuronSynapse> Synapses;

		TSpikeNeuronImplInner Inner;
		TNeuronSpaceInfo SpaceInfo;

		TOptional<typename TConf::TNeuronSynapse::TConst::TProto> PredefineSynapseConst;
	};

} // namespace NDnn