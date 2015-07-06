#pragma once


#include <dnn/io/stream.h>
#include <dnn/util/spinning_barrier.h>
#include <dnn/neurons/spike_neuron.h>
#include <dnn/base/constants.h>

#include "builder.h"
#include "network.h"
#include "global_ctx.h"
#include "sim_info.h"

namespace dnn {


class Sim : public Printable {
public:
	Sim(const Constants &_c) : c(_c), duration(0.0) {
		GlobalCtx::inst().init(sim_info, c);
	}
	
	void build(Stream* input_stream = nullptr);
	
	void serialize(Stream &output_stream);
	void saveStat(Stream &str);
	void saveSpikes(Stream &str);
	void turnOnStatistics();
	
	static void runWorker(Sim &s, size_t from, size_t to, SpinningBarrier &barrier, std::exception_ptr &eptr);
	static void runWorkerRoutine(Sim &s, size_t from, size_t to, SpinningBarrier &barrier);
	
	void setMaxDuration(const double Tmax);
	void print(std::ostream &str) const;
	void run(size_t jobs);

protected:

	SimInfo sim_info;
	double duration;
	const Constants &c;
	vector<InterfacedPtr<SpikeNeuronBase>> neurons;
	uptr<Network> net;
};

}
