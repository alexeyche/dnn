#pragma once

#include <dnn/protos/config.pb.h>
#include <dnn/util/optional.h>
#include <dnn/sim/sim.h>
#include <dnn/util/protobuf.h>
#include <dnn/util/serial/bin_serial.h>

namespace NDnn {

	struct TModelOptions {
		ui32 Port;
		TOptional<ui32> Jobs;
		TOptional<TString> ConfigFile;
		TOptional<TString> ModelLoad;
		TOptional<TString> ModelSave;
		TOptional<TString> InputSpikesFile;
		TOptional<TString> InputTimeSeries;
		TOptional<TString> OutputSpikesFile;
		TOptional<TString> StatFile;
		TString Name;
	};

	TModelOptions InitOptions(const int argc, const char** argv, TString name, std::set<int> fields = {});

	template <typename ... T>
	auto BuildModel(TModelOptions options) {
		auto sim = BuildSim<T...>(options.Port);

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
	    	L_DEBUG << "Reading input spikes " << *options.InputSpikesFile;
    		std::ifstream input(*options.InputSpikesFile, std::ios::binary);
		    TBinSerial serial(input);
	    	sim.SetInputSpikes(serial.ReadObject<TSpikesList>());
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
	    return sim;
	}

} // namespace NDnn