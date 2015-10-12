#pragma once


#include <dnn/io/stream.h>
#include <dnn/util/spinning_barrier.h>
#include <dnn/neurons/spike_neuron.h>
#include <dnn/base/constants.h>

#include "reward_control.h"
#include "builder.h"
#include "network.h"
#include "global_ctx.h"
#include "sim_info.h"

namespace dnn {


class Sim : public Printable {
public:
	Sim() : duration(0.0) {
		init();
	}

	Sim(const Constants &_c) : c(_c) {
		init();
	}

	void build(Stream* input_stream = nullptr);

	void serialize(Stream &output_stream);
	void saveStat(Stream &str);
	void saveSpikes(Stream &str);
	void turnOnStatistics();

	static void runWorker(Sim &s, size_t from, size_t to, SpinningBarrier &barrier, bool master_thread, std::exception_ptr &eptr);
	static void runWorkerRoutine(Sim &s, size_t from, size_t to, SpinningBarrier &barrier, bool master_thread);

	void setMaxDuration(const double Tmax);
	void print(std::ostream &str) const;
	void run(size_t jobs);

protected:
	void init() {
		GlobalCtx::inst().init(sim_info, c, duration, rc);
	}
	SimInfo sim_info;

	RewardControl rc;

	double duration;
	Constants c;
	vector<InterfacedPtr<SpikeNeuronBase>> neurons;
	uptr<Network> net;
};

}
