#pragma once

#include "layer.h"
#include "network.h"
#include "global_ctx.h"
#include "reward_control.h"

#include <dnn/base/base.h>

#include <dnn/dispatcher/dispatcher.h>

#include <dnn/util/log/log.h>
#include <dnn/util/rand.h>
#include <dnn/util/serial/bin_serial.h>
#include <dnn/util/spinning_barrier.h>
#include <dnn/util/thread.h>
#include <dnn/util/tuple.h>
#include <dnn/util/stat_gatherer.h>
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
			serial(Port);
			serial(Seed);
			serial(PastTime);
		}

		ui32 Jobs = 4;
		double Duration = 1000;
		double Dt = 1.0;
		ui32 Port = 9090;
		int Seed = -1;
		double PastTime = 0.0;
	};


	template <typename ... T>
	class TSim: public IProtoSerial<NDnnProto::TConfig> {
	public:
		using TSelf = TSim<T...>;
		using TParent = IProtoSerial<NDnnProto::TConfig>;

		TSim(const TModelOptions& options)
			: Dispatcher(options.Port)
			, PopulationSize(0)
			, Options(options)
		{
			ForEachEnumerate(Layers, [&](ui32 layerId, auto& l) {
				l.SetupSpaceInfo(layerId, PopulationSize);
				PopulationSize += l.Size();
			});

			TGlobalCtx::Inst().Init(RewardControl);
			TGlobalCtx::Inst().SetPastTime(Conf.PastTime);
		}

		TSim(const TSim& other)
			: Dispatcher(other.Dispatcher.GetPort())
		{
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
				TGlobalCtx::Inst().Init(RewardControl);
				TGlobalCtx::Inst().SetPastTime(Conf.PastTime);
				Network = other.Network;
				Network.Init(PopulationSize);
				ForEach(Layers, [&](auto& l) {
					Network.AddLayer(l);
				});
			}
			return *this;
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

		void SetDuration(double duration) {
			Conf.Duration = duration;
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
				Dispatcher.SetPort(Conf.Port);
				const NDnnProto::TConfig& inputConfig = serial.GetMessage<NDnnProto::TConfig>();
				CreateConnections(inputConfig);

				Network.Init(PopulationSize);
				ForEach(Layers, [&](auto& l) {
					Network.AddLayer(l);
				});
			}
            serial(RewardControl,  NDnnProto::TConfig::kRewardControlFieldNumber);
		}

		void SetInputSpikes(const TSpikesList&& ts) {
			Network.GetMutSpikesList().Info = ts.Info;
			auto& firstLayer = std::get<0>(Layers);
			ENSURE(ts.Dim() == firstLayer.Size(), "Size of input spikes list doesn't equal to first layer size: " << ts.Dim() << " != " << firstLayer.Size());

			for (auto& n: firstLayer) {
				n.GetNeuron().SetSpikeSequence(ts[n.GetLocalId()]);
			}
			Conf.Duration = ts.Info.GetDuration();
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
			Conf.Duration = ts.Length();
			Network.GetMutSpikesList().Info = ts.Info;
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
