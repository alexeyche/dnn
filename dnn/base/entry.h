#pragma once

#include "model_options.h"

#include <dnn/protos/config.pb.h>
#include <dnn/sim/sim.h>
#include <ground/protobuf.h>
#include <ground/serial/bin_serial.h>

namespace NDnn {

	TModelOptions InitOptions(const int argc, const char** argv, TString name, std::set<int> fields = {});

	template <typename ... T>
	auto BuildModel(TModelOptions options = TModelOptions()) {
		auto sim = BuildSim<T...>(options);
		
		if (options.Seed) {
	    	sim.SetSeed(*options.Seed);
	    }
		if (options.ConnectionSeed) {
	    	sim.SetConnectionSeed(*options.ConnectionSeed);
	    }

		if (options.ConfigFile) {
			L_DEBUG << "Reading config " << *options.ConfigFile;
	    	NDnnProto::TConfig config;
	    	ReadProtoTextFromFile(*options.ConfigFile, config);
	    	sim.Deserialize(config);
	    }

		if (options.ModelLoad) {
			L_DEBUG << "Reading model " << *options.ModelLoad;
			std::ifstream input(*options.ModelLoad, std::ios::binary);
			TBinSerial serial(input);
			NDnnProto::TConfig proto;
			serial.ReadProtobufMessage(proto);
			sim.Deserialize(proto);
		}

	    if (options.InputSpikesFile) {
	    	sim.SetInputSpikes(*options.InputSpikes);
	    }

	    if (options.InputTimeSeries) {
	    	L_DEBUG << "Reading input time series " << *options.InputTimeSeries;
    		std::ifstream input(*options.InputTimeSeries, std::ios::binary);
		    TBinSerial serial(input);
	    	sim.SetInputTimeSeries(serial.ReadObject<TTimeSeries>());
	    }

	    if (options.Jobs) {
	    	sim.SetJobs(*options.Jobs);
	    }
	    if (options.Tmax) {
	    	sim.SetDuration(*options.Tmax);
	    }
	    return sim;
	}

} // namespace NDnn