#include "sim.h"


#include <dnn/util/log/log.h>

namespace dnn {


void Sim::build(Stream* input_stream) {
	global_neuron_index = 0;

	Builder b(c);
	if(input_stream) {
		auto sim_info_serial = input_stream->readDynamic<SimInfo>();
		sim_info = sim_info_serial.ref();
		sim_info_serial.destroy();

		auto rc_serial = input_stream->readDynamic<RewardControl>();
		rc = rc_serial.ref();
		rc_serial.destroy();

		b.setInputModelStream(input_stream);
	} else {
		rc = b.buildRewardControlFromConstants();
	}

	neurons = b.buildNeurons();
	for(auto &n: neurons) {
		duration = std::max(duration, n.ref().getSimDuration());
	}
	net = uptr<Network>(new Network(neurons));
	if(b.getInputFileNames().size()>0) {
		net->spikesList().ts_info = b.getInputTimeSeriesInfo();
	} else {
		TimeSeriesInfo default_info;
		default_info.addLabelAtPos("default", duration);
		net->spikesList().ts_info = default_info;
	}
}

void Sim::serialize(Stream &output_stream) {
	output_stream.writeObject(&sim_info);
	output_stream.writeObject(&rc);

	for(auto &n : neurons) {
		output_stream.writeObject(n.ptr());
	}
}
void Sim::saveStat(Stream &str) {
	for(auto &n: neurons) {
		if(n.ref().getStat().on()) {
			Statistics st = n.ref().getStat();
			str.writeObject(&st);
		}
	}
	str.writeObject(&rc.getStat());
}
void Sim::saveSpikes(Stream &str) {
	if(!net.get()) {
		throw dnnException()<< "Sim network was not found. You need to build sim\n";
	}
	str.writeObject(&net->spikesList());
}

void Sim::turnOnStatistics() {
	Builder::turnOnStatistics(neurons, c.sim_conf.neurons_to_listen);
	rc.getStat().turnOn();
}

void Sim::runWorkerRoutine(Sim &s, size_t from, size_t to, SpinningBarrier &barrier, bool master_thread) {
	Time t(s.c.sim_conf.dt);

	for(size_t i=from; i<to; ++i) {
		s.neurons[i].ref().resetInternal();
	}


	barrier.wait();
	L_DEBUG << "Dive in main loop for neurons from " << from << " to " << to;

	#ifdef PERF
	std::time_t start_time = std::time(nullptr);
	double sim_time = t.t;
	#endif

	for(; t<s.duration; ++t) {
		// L_DEBUG << "[Layer of neurons " << from << ":" << to << "] Tick at time " << t.t;

		if(master_thread) GlobalCtx().inst().setCurrentClassId(s.net->getClassId(t));

		barrier.wait();
		for(size_t i=from; i<to; ++i) {
			// L_DEBUG << "[Layer of neurons " << from << ":" << to << "] Simulating neuron " << i;
			s.neurons[i].ref().calculateDynamicsInternal(t);

			if(s.neurons[i].ref().fired()) {
				// L_DEBUG << "\t[Layer of neurons " << from << ":" << to << "] Spiked " << s.neurons[i].ref().id() << " at " << t.t;
				s.net->propagateSpike(s.neurons[i].ref(), t.t);
				s.neurons[i].ref().setFired(false);
			}
		}
		barrier.wait();
		if(master_thread) s.rc.calculateDynamics(t);

		#ifdef PERF
		size_t cur_time = std::time(nullptr);
		if(cur_time - start_time>5) {
			L_DEBUG << "Sim, perf start: " << ((double)(t.t-sim_time)/1000.0)/((double)(cur_time - start_time));
			start_time = cur_time;
			sim_time = t.t;
		}
		#endif
	}
	L_DEBUG << "Main loop for neurons from " << from << " to " << to << " is finished";
	barrier.wait();
}


void Sim::runWorker(Sim &s, size_t from, size_t to, SpinningBarrier &barrier, bool master_thread, std::exception_ptr &eptr) {
	try {
		runWorkerRoutine(s, from, to, barrier, master_thread);
	} catch (const dnnException &e) {
		L_DEBUG << "Got error in [" << from << ":" << to << "] thread: " << e.what();
		eptr = std::current_exception();
		barrier.fail();
	} catch (const dnnInterrupt &e) {
		// pass
	}
}

void Sim::run(size_t jobs) {
	if(fabs(duration) < 0.00001) {
		throw dnnException() << "Duration of simulation is " << duration << ". Check that input data was provided\n";
	}
	L_DEBUG << "Going to run simulation for " << duration << " ms in " << jobs << " jobs";
	vector<IndexSlice> slices = dispatchOnThreads(neurons.size(), jobs);
	vector<std::thread> threads;
	vector<std::exception_ptr> exceptions;

	SpinningBarrier barrier(jobs);
	for(auto &slice: slices) {
		exceptions.emplace_back();
		threads.emplace_back(
			Sim::runWorker,
			std::ref(*this),
			slice.from,
			slice.to,
			std::ref(barrier),
			slice.from == 0,
			std::ref(exceptions.back())
		);
	}

	for(auto &t: threads) {
		t.join();
	}
	for(auto eptr: exceptions) {
		try {
			if(eptr) {
				std::rethrow_exception(eptr);
			}
		} catch(const dnnException &dnn_e) {
			throw;
		}
	}
	sim_info.pastTime += duration;
}

void Sim::setMaxDuration(const double Tmax) {
	if(fabs(duration) < 0.00001) {
		throw dnnException() << "Setting max duration for empty sim\n";
	}
	duration = min(Tmax, duration);
}
void Sim::print(std::ostream &str) const {
	str << "Sim\n";
	str << "  " << neurons.size() << " neurons ready to simulate for " << duration << "ms\n";
}

}