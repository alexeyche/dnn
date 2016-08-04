#pragma once

#include <dnn/neuron/spike_sequence_neuron.h>

namespace NDnn {

	template <typename ... T>
	void TSim<T...>::Run() {
		L_DEBUG << "Going to run simulation of " << LayersSize() << " layers, for " << Conf.Duration << " ms in " << Conf.Jobs << " jobs";
		
		TVector<TIndexSlice> perLayerJobs = DispatchOnThreads(Conf.Jobs, LayersSize());
		
		// Hack for spending one thread for SpikeSequence neuron
		if (std::is_same<typename std::tuple_element<0,decltype(Layers)>::type::TNeuronType, TSpikeSequenceNeuron>::value) {
			ui32 firstLayerJobs = perLayerJobs[0].Size;
			perLayerJobs[0].From = 0;
			perLayerJobs[0].To = 1;
			perLayerJobs[0].Size = 1;
			ENSURE(perLayerJobs.size() > 1, "It is strange to simulate only SpikeSequence neuron");
			perLayerJobs[1].From -= firstLayerJobs-1;
			perLayerJobs[1].Size += firstLayerJobs-1;
		}
	 	
	 	TSpinningBarrier barrier(Conf.Jobs);
		TVector<std::thread> threads;
		std::mutex errorsMut;
		TVector<std::exception_ptr> errors;
		
		ENSURE(Conf.Jobs >= LayersSize(), "Need jobs number same or bigger than number of layers");

		ForEachEnumerate(Layers, [&](ui32 layerId, auto& layer) {
			SimLayer(layer, perLayerJobs[layerId].Size, threads, errors, errorsMut, barrier, layerId == 0 ? true : false);
		});
		
		std::thread dispatcherThread = std::thread([&]() {
			if (Options.Port) {
				Dispatcher.SetPort(*Options.Port);
				Dispatcher.MainLoop();	
			}
		});
		
		for (auto& t: threads) {
			t.join();
		}
		Dispatcher.ShutDown();
		dispatcherThread.join();

		for (const auto& err: errors) {
			std::rethrow_exception(err);
		}

		Conf.PastTime += Conf.Duration;

		if (Options.OutputSpikesFile) {
			L_DEBUG << "Saving spikes in " << *Options.OutputSpikesFile;
        	SaveSpikes(*Options.OutputSpikesFile);
        }
        if (Options.StatFile) {
        	L_DEBUG << "Saving statistics in " << *Options.StatFile;
           	SaveStat(*Options.StatFile);
        }
        if (Options.ModelSave) {
        	L_DEBUG << "Saving model in " << *Options.ModelSave;
            SaveModel(*Options.ModelSave);
        }
	}


	template <typename ...T>
	template <typename L>
	void TSim<T...>::SimLayer(L& layer, ui32 jobs, TVector<std::thread>& threads, TVector<std::exception_ptr>& errors, std::mutex& errorsMut, TSpinningBarrier& barrier, bool masterThread) {
		if (std::is_same<typename L::TNeuronType, TSpikeSequenceNeuron>::value) {
			threads.emplace_back(
				TSelf::RunWorker<L>,
				std::ref(*this),
				std::ref(layer),
				0,
				layer.Size(),
				std::ref(barrier),
				masterThread ? true : false,
				std::ref(errors),
				std::ref(errorsMut)
			);
		} else {
			TVector<TIndexSlice> layerJobSlices = DispatchOnThreads(layer.Size(), jobs);
			for (ui32 sliceId = 0; sliceId < layerJobSlices.size(); ++sliceId) {
				const TIndexSlice& slice = layerJobSlices[sliceId];
				threads.emplace_back(
					TSelf::RunWorker<L>,
					std::ref(*this),
					std::ref(layer),
					slice.From,
					slice.To,
					std::ref(barrier),
					(masterThread && (sliceId == 0)) ? true : false,
					std::ref(errors),
					std::ref(errorsMut)
				);
			}	
		}
	}

	template <typename ...T>
	double TSim<T...>::GetInputFromDispatcher(ui32 layerId, ui32 neuronId) {
		return Dispatcher.GetNeuronInput(layerId, neuronId);
	}

	template <typename ...T>
	template <typename L>
	void TSim<T...>::RunWorker(TSelf& self, L& layer, ui32 idxFrom, ui32 idxTo, TSpinningBarrier& barrier, bool masterThread, TVector<std::exception_ptr>& errors, std::mutex& errorsMut) {
		try {
			self.RunWorkerRoutine<L>(layer, idxFrom, idxTo, barrier, masterThread);
		} catch (const TErrException& e) {
			L_ERROR << "Got error in layer " << layer.GetId() << ", neurons " << idxFrom << ":" << idxTo << ", thread: " << e.what();
			barrier.Fail();
			TGuard guard(errorsMut);
			errors.emplace_back(std::current_exception());
		} catch (const TErrInterrupt& e) {
			// pass
		}
	}


	template <typename ... T>
	template <typename L>
	void TSim<T...>::RunWorkerRoutine(L& layer, ui32 idxFrom, ui32 idxTo, TSpinningBarrier& barrier, bool masterThread) {
		TTime t(Conf.Dt);
		TRandEngine rand(Conf.Seed);
		if (masterThread) StatGatherer.Init();

		L_DEBUG << "Entering into simulation of layer " << layer.GetId() << " of neurons " << idxFrom << ":" << idxTo;

		for (ui32 neuronId=idxFrom; neuronId<idxTo; ++neuronId) {
			layer[neuronId].SetRandEngine(rand);
			layer[neuronId].Prepare();
		}


//======= PERFOMANCE MEASURE ======================================================================
#ifdef PERF

		std::time_t start_time = std::time(nullptr);
		double sim_time = t.T;

#endif
//======= END =====================================================================================
		barrier.Wait();

		for (; t < Conf.Duration; ++t) {
			if (masterThread) TGlobalCtx().Inst().SetCurrentClassId(Network.GetClassId(t));
			barrier.Wait();

			for(ui32 neuronId=idxFrom; neuronId<idxTo; ++neuronId) {
				double input = 0.0;
				if (layer.HasInput()) {
					input = Dispatcher.GetNeuronInput(layer.GetId(), neuronId);
				}
				layer[neuronId].CalculateDynamicsInternal(t, input);
				if (layer[neuronId].GetNeuron().Fired()) {
					Network.PropagateSpike(layer[neuronId], t);
					layer[neuronId].GetNeuron().PostSpikeDynamics(t);
					layer[neuronId].GetNeuron().MutFired() = false;
				}
			}

			barrier.Wait();
			if (masterThread) { 
				StatGatherer.Collect(t);
				RewardControl.CalculateDynamics(t);
			}
			barrier.Wait();

//======= PERFOMANCE MEASURE ======================================================================
#ifdef PERF

			size_t cur_time = std::time(nullptr);
			if(cur_time - start_time>5) {
				L_DEBUG << "Sim, perf start: " << ((double)(t.T-sim_time)/1000.0)/((double)(cur_time - start_time)) << " " << (double)(t.T-sim_time)/1000.0 << " / " << (double)(cur_time - start_time);
				start_time = cur_time;
				sim_time = t.T;
			}

#endif
//======= END =====================================================================================

		}

		barrier.Wait();
	}


	template <typename ...T>
	void TSim<T...>::CreateConnections(const NDnnProto::TConfig& config) {
		TRandEngine rand(Conf.ConnectionSeed);
		for (const auto& connection: config.connection()) {
			ForEach(Layers, [&](auto& leftLayer) {
				if (leftLayer.GetId() != connection.from()) {
					return;
				}
				ForEach(Layers, [&](auto& rightLayer) {
					if (rightLayer.GetId() != connection.to()) {
						return;
					}
					L_DEBUG << "Connecting layer " << leftLayer.GetId() << " to " << rightLayer.GetId();
					leftLayer.Connect(rightLayer, connection, rand);
				});
			});
		}
		ForEach(Layers, [&](auto& layer) {
			for (auto& n: layer) {
				n.InitReceptiveField(rand);
			}
		});
	}


} // namespace NDnn
