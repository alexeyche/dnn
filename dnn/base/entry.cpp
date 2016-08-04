#include "entry.h"

#include <ground/string.h>
#include <ground/proto_options.h>
#include <dnn/protos/options.pb.h>

#include <cstdlib>

namespace NDnn {

	TModelOptions InitOptions(const int argc, const char** argv, TString name, std::set<int> fields) {
		TProtoOptions<NDnnProto::TDnnOptions> clOptions(argc, argv, NStr::TStringBuilder() << "Dynamic neural network model, " << name, fields);
		NDnnProto::TDnnOptions options;
		if (!clOptions.Parse(options)) {
			exit(1);
		}

	    if (options.verbose()) {
	        TLog::Instance().SetLogLevel(TLog::DEBUG_LEVEL);
	    }

	    TModelOptions opts;
	    opts.Name = name;

	    if (options.has_config()) {
	    	opts.ConfigFile = options.config();
	    }

	    if (options.has_inputspikes()) {
	    	opts.InputSpikesFile = options.inputspikes();
	    	L_DEBUG << "Reading input spikes " << *opts.InputSpikesFile;
    		std::ifstream input(*opts.InputSpikesFile, std::ios::binary);
		    TBinSerial serial(input);
		    opts.InputSpikes = serial.ReadObject<TSpikesList>();
	    }

	    if (options.has_targetspikes()) {
	    	opts.TargetSpikesFile = options.targetspikes();
	    	L_DEBUG << "Reading target spikes " << *opts.TargetSpikesFile;
    		std::ifstream input(*opts.TargetSpikesFile, std::ios::binary);
		    TBinSerial serial(input);
		    opts.TargetSpikes = serial.ReadObject<TSpikesList>();
	    }

	    if (options.has_inputtimeseries()) {
	    	opts.InputTimeSeries = options.inputtimeseries();
	    }

	    if (options.has_port()) {
	        opts.Port = options.port();
	    }

	    if (options.has_output()) {
	    	opts.OutputSpikesFile = options.output();
	    }

	    if (options.has_stat()) {
	    	opts.StatFile = options.stat();
	    }

	    if (options.has_jobs()) {
	    	opts.Jobs = options.jobs();
	    }
	    
	    if (options.has_load()) {
	    	opts.ModelLoad = options.load();
	    }
	    
	    if (options.has_save()) {
	    	opts.ModelSave = options.save();
	    }
	    
	    if (options.has_tmax()) {
	    	opts.Tmax = options.tmax();
	    }

	    if (options.has_nolearning()) {
	    	opts.NoLearning = true;
	    }

	    if (options.has_seed()) {
			opts.Seed = options.seed();
		}

    	if (options.has_connectionseed()) {
			opts.ConnectionSeed = options.connectionseed();
		}
		
	    return opts;
	}


} // namespace NDnn