#pragma once

#include <queue>
#include <atomic>

#include <dnn/util/act_vector.h>
#include <dnn/util/random.h>
#include <dnn/util/rand.h>
#include <dnn/util/optional.h>
#include <dnn/util/serial/meta_proto_serial.h>
#include <dnn/util/serial/util.h>
#include <dnn/protos/config.pb.h>
#include <dnn/protos/spike_neuron_impl.pb.h>


namespace NDnn {




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

	template <typename RF>
	void CallInitReceptiveField(RF& rf, const TNeuronSpaceInfo& info) {
		rf.Init(info);
	}

	template <>
	void CallInitReceptiveField<TEmpty>(TEmpty&, const TNeuronSpaceInfo&);

	template <typename RF>
	double CallCalculateResponseReceptiveField(RF& rf, double I) {
		return rf.CalculateResponse(I);
	}

	template <>
	double CallCalculateResponseReceptiveField<TEmpty>(TEmpty&, double);

	template <typename R, typename N>
	struct TCallPrepareReinforcement {
		void operator ()(R& r, N& neuron) {
			r.SetNeuronImpl(neuron);
		}
	};
	template <typename N>
	struct TCallPrepareReinforcement<TEmpty, N> {
		void operator ()(TEmpty&, N&) {}
	};


	template <typename R>
	void CallModulateRewardReinforcement(R& r) {
		r.ModulateReward();
	}

	template <>
	void CallModulateRewardReinforcement<TEmpty>(TEmpty&);

	
	template <typename TNeuron, typename TConf>
	class TSpikeNeuronImpl: public IMetaProtoSerial {
	public:
		using TNeuronType = TNeuron;
		using TSelf = TSpikeNeuronImpl<TNeuron, TConf>;

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
		    	if (fabs(x) < 0.0001) {
		        	Synapses.SetInactive(synIdIt);
		        } else {
			    	Isyn += x;
		        	++synIdIt;
		        }
		    }
			double Irf = CallCalculateResponseReceptiveField(ReceptiveField, Iinput);
			// ENSURE(!std::isnan(Isyn), "Isyn is nan");
			Neuron.CalculateDynamics(t, Irf, Isyn);

		    Neuron.MutSpikeProbability() = Activation.SpikeProbability(Neuron.Membrane());
			if(t.Dt * Neuron.SpikeProbability() > Rand->GetUnif()) {
		        Neuron.MutFired() = true;
		        Neuron.PostSpikeDynamics(t);
		    }

	    	LearningRule.CalculateDynamics(t);
			LearningRule.MutNorm().CalculateDynamics(t);
	
		    CallModulateRewardReinforcement(Reinforcement);

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
		}

		void Prepare() {
			ENSURE(Rand, "Random engine is not set");
			CallInitReceptiveField<typename TConf::TNeuronReceptiveField>(ReceptiveField, SpaceInfo);
			LearningRule.SetNeuronImpl(*this);
			LearningRule.Reset();
			TCallPrepareReinforcement<typename TConf::template TNeuronReinforcement<TSelf>, TSelf>()(Reinforcement, *this);
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

	private:
		TPtr<TRandEngine> Rand;

		TAsyncSpikeQueue Queue;

		TNeuron Neuron;

		typename TConf::TNeuronActivationFunction Activation;
		typename TConf::TNeuronReceptiveField ReceptiveField;
		typename TConf::template TNeuronLearningRule<TSelf> LearningRule;
		typename TConf::template TNeuronReinforcement<TSelf> Reinforcement;

		TActVector<typename TConf::TNeuronSynapse> Synapses;

		TSpikeNeuronImplInner Inner;
		TNeuronSpaceInfo SpaceInfo;

		TOptional<typename TConf::TNeuronSynapse::TConst::TProto> PredefineSynapseConst;
	};

} // namespace NDnn