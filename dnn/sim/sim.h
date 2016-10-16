#pragma once

#include "layer.h"
#include "network.h"
#include "global_ctx.h"
#include "reward_control.h"

#include <ground/base/base.h>

#include <dnn/dispatcher/dispatcher.h>

#include <ground/log/log.h>
#include <ground/rand.h>
#include <ground/serial/bin_serial.h>
#include <ground/spinning_barrier.h>
#include <ground/thread.h>
#include <ground/tuple.h>
#include <ground/stat_gatherer.h>
#include <dnn/base/model_options.h>

#include <dnn/neuron/integrate_and_fire.h>

#include <dnn/protos/config.pb.h>

#include <utility>

namespace NDnn {

	struct TSimConfiguration: public IProtoSerial<NDnnProto::TSimConfiguration> {
		void SerialProcess(TProtoSerial& serial) override {
			serial(Jobs);
			serial(Duration);
			serial(Dt);
			serial(Seed);
			serial(ConnectionSeed);
			serial(PastTime);
		}

		ui32 Jobs = 4;
		double Duration = 1000;
		double Dt = 1.0;
		int Seed = -1;
		int ConnectionSeed = -1;
		double PastTime = 0.0;
	};


	template <typename ... T>
	class TSim: public IProtoSerial<NDnnProto::TConfig> {
	public:
		using TSelf = TSim<T...>;
		using TParent = IProtoSerial<NDnnProto::TConfig>;

		TSim(const TModelOptions& options)
			: PopulationSize(0)
			, Options(options)
		{
			TVector<ui32> sizeOfLayers;
			ForEachEnumerate(Layers, [&](ui32 layerId, auto& l) {
				l.SetupSpaceInfo(layerId, PopulationSize);
				PopulationSize += l.Size();
				sizeOfLayers.push_back(l.Size());
			});

			TGlobalCtx::Inst().Init(RewardControl, sizeOfLayers, TVector<TVector<TDestinationInfo>>(PopulationSize));
			TGlobalCtx::Inst().SetPastTime(Conf.PastTime);
		}

		TSim(const TSim& other) {
			(*this) = other;
		}

		TSim& operator = (const TSim& other) {
			if (this != &other) {
				ENSURE(other.StatGatherer.Size() == 0, "Can't copy simulation with pointed variables to listen in statistics gatherer");
				Layers = other.Layers;
				PopulationSize = other.PopulationSize;
				Conf = other.Conf;
				Options = other.Options;
				Dispatcher = other.Dispatcher;
				RewardControl = other.RewardControl;
				Network = other.Network;
				Network.Init(PopulationSize);
				TVector<ui32> sizeOfLayers;
				ForEach(Layers, [&](auto& l) {
					Network.AddLayer(l);
					sizeOfLayers.push_back(l.Size());
				});
				TGlobalCtx::Inst().Init(RewardControl, sizeOfLayers, TGlobalCtx::Inst().GetAdjacentNeuronInfo());
				TGlobalCtx::Inst().SetPastTime(Conf.PastTime);
			}
			return *this;
		}

		const ui32 GetPopulationSize() const {
			return PopulationSize;
		}

		template <size_t layerId, size_t neuronId>
		auto GetNeuron() {
			return std::get<layerId>(Layers)[neuronId].GetNeuron();
		}

		template <size_t layerId, size_t neuronId>
		auto GetLearningRule() {
			return std::get<layerId>(Layers)[neuronId].GetLearningRule();
		}

		template <size_t layerId, size_t neuronId, size_t synapseId>
		auto GetSynapse() {
			const auto& synVec = std::get<layerId>(Layers)[neuronId].GetSynapses();
			return synVec.at(synapseId);
		}

		template <size_t layerId>
		const auto& GetLayer() const {
			return std::get<layerId>(Layers);
		}

		template <size_t layerId>
		auto& GetMutLayer() {
			return std::get<layerId>(Layers);
		}


		void ListenStat(const TString& name, std::function<double()> cb, ui32 from, ui32 to) {
			StatGatherer.ListenStat(name, cb, from, to);
		}

		void CollectReward() {
			StatGatherer.ListenStat("Reward", [&]() { return TGlobalCtx::Inst().GetReward(); }, 0, Conf.Duration);	
			StatGatherer.ListenStat("RewardDelta", [&]() { return TGlobalCtx::Inst().GetRewardDelta(); }, 0, Conf.Duration);
		}
 
		template <size_t layerId, size_t neuronId>
		void ListenBasicStats(ui32 from, ui32 to) {
			StatGatherer.ListenStat("Membrane", [&]() { return GetNeuron<layerId, neuronId>().Membrane(); }, from, to);
			StatGatherer.ListenStat("SpikeProbability", [&]() { return GetNeuron<layerId, neuronId>().SpikeProbability(); }, from, to);
		}

		void SaveStat(const TString& fname) {
			StatGatherer.SaveStat(fname);
		}

 		void SaveModel(const TString& fname) {
			std::ofstream output(fname, std::ios::binary);
		    TBinSerial serial(output);
		    NDnnProto::TConfig config = Serialize();
		    serial(config, EProto::CONFIG);
		}

		void SetJobs(ui32 jobs) {
			Conf.Jobs = jobs;
		}

		void SetSeed(ui32 seed) {
			Conf.Seed = seed;
		}

		void SetConnectionSeed(ui32 seed) {
			Conf.ConnectionSeed = seed;
		}

		void SetDuration(double duration) {
			Conf.Duration = duration;
		}

		const double& GetDuration() const {
			return Conf.Duration;
		}

		ui32 LayersSize() const {
			return std::tuple_size<decltype(Layers)>::value;
		}

		void SerialProcess(TProtoSerial& serial) {
			serial(Conf, NDnnProto::TConfig::kSimConfigurationFieldNumber);

			serial.DuplicateSingleRepeated(NDnnProto::TConfig::kLayerFieldNumber, LayersSize());
			ForEach(Layers, [&](auto& layer) {
				serial(layer, NDnnProto::TConfig::kLayerFieldNumber, /* newMessage = */ true);
			});
			if (serial.IsInput()) {
				const NDnnProto::TConfig& inputConfig = serial.GetMessage<NDnnProto::TConfig>();
				CreateConnections(inputConfig);

				Network.Init(PopulationSize);
				ForEach(Layers, [&](auto& l) {
					Network.AddLayer(l);
				});
			}
            serial(RewardControl,  NDnnProto::TConfig::kRewardControlFieldNumber);
		}

		void SetInputSpikes(const TSpikesList& ts) {
			Network.GetMutSpikesList().Info = ts.Info;
			auto& firstLayer = std::get<0>(Layers);
			ENSURE(ts.Dim() == firstLayer.Size(), "Size of input spikes list doesn't equal to first layer size: " << ts.Dim() << " != " << firstLayer.Size());

			double max_spike = std::numeric_limits<double>::min();
			for (auto& n: firstLayer) {
				const TVector<double>& spikes = ts[n.GetLocalId()];
				n.GetNeuron().SetSpikeSequence(spikes);
				if (spikes.size() > 0) {
					max_spike = std::max(max_spike, spikes.back());	
				}
			}
			Conf.Duration = ts.Info.GetDuration();
			if (Conf.Duration == 0.0) {
				Conf.Duration = max_spike;
			}
		}

		bool RequireInput() {
			bool requireInput = false;
			ForEach(Layers, [&](const auto& layer) {
				if (layer.HasInput()) {
					requireInput = true;;
				}
			});
			return requireInput;
		}

		void SetInputTimeSeries(const TTimeSeries&& ts) {
			ui32 requiredDimSize = 0;
			ForEach(Layers, [&](auto& layer) {
				if (layer.HasInput()) {
					requiredDimSize += layer.Size();	
				}
			});
			Conf.Duration = ts.Length() * Conf.Dt;

			Network.GetMutSpikesList().Info = ts.Info;
			for (auto& lab: Network.GetMutSpikesList().Info.Labels) {
				lab.From = lab.From * Conf.Dt;
				lab.To = lab.To * Conf.Dt;
			}
			if (ts.Dim() == 1) {
				L_DEBUG << "Got one dimensional time series, dnn will duplicate data on " << requiredDimSize << " dimensions";
				TTimeSeries dupTs(ts);
				dupTs.MultiplyOnDimensions(requiredDimSize);
				Dispatcher.SetInputTimeSeries(std::forward<const TTimeSeries>(dupTs));
			} else {
				ENSURE(ts.Dim() == requiredDimSize, "Input time series is not statisfy to input layer size: " << ts.Dim() << " != " << requiredDimSize);
				Dispatcher.SetInputTimeSeries(std::forward<const TTimeSeries>(ts));	
			}
			
		}

		const TSpikesList& GetSpikes() const {
			return Network.GetSpikesList();
		}

		void SaveSpikes(const TString& fname) const {
			std::ofstream output(fname, std::ios::binary);
	    	TBinSerial serial(output);
			serial.WriteObject<TSpikesList>(Network.GetSpikesList());
		}

		void Run();
		
	private:
		double GetInputFromDispatcher(ui32 layerId, ui32 neuronId);

		template <typename L>
		void RunWorkerRoutine(L& layer, ui32 idxFrom, ui32 idxTo, TSpinningBarrier& barrier, bool masterThread);

		template <typename L>
		void SimLayer(L& layer, ui32 jobs, TVector<std::thread>& threads, TVector<std::exception_ptr>& errors, std::mutex& errorsMut, TSpinningBarrier& barrier, bool masterThread);

		template <typename L>
		static void RunWorker(TSelf& self, L& layer, ui32 idxFrom, ui32 idxTo, TSpinningBarrier& barrier, bool masterThread, TVector<std::exception_ptr>& errors, std::mutex& errorsMut);

		void CreateConnections(const NDnnProto::TConfig& config);

	private:
		std::tuple<T ...> Layers;

		TSimConfiguration Conf;
		TDispatcher Dispatcher;
		TNetwork Network;

		ui32 PopulationSize;
		TStatGatherer StatGatherer;

		TRewardControl RewardControl;
		TModelOptions Options;
	};


	template <typename ... T>
	auto BuildSim(const TModelOptions& options) {
		return TSim<T...>(options);
	}

} // namespace NDnn

#include "sim_impl.h"
