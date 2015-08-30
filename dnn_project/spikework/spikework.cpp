
#include "spikework.h"
#include "fft.h"
#include "conv.h"
#include "kernel.h"
#include "gram.h"
#include "read.h"

#include <dnn/util/log/log.h>

namespace dnn {


Spikework::~Spikework() {
    for(auto &v: workers) {
        delete v.ptr();
    }
}

Spikework::Spikework(const vector<string> &args) {
	work_map["read"] = &createWorker<ReadWorker>;
	work_map["fft"] = &createWorker<FFTWorker>;
	work_map["conv"] = &createWorker<ConvWorker>;
	work_map["kernel"] = &createWorker<KernelWorker>;
	work_map["gram"] = &createWorker<GramWorker>;

	vector<string> spikework_args;
	for(const auto &a: args) {
		auto p_ptr = work_map.find(a);
		if(p_ptr != work_map.end()) {
			break;
		}
		spikework_args.push_back(a);
	}

	bool need_help = false;
	bool verbose = false;
	bool no_colors = false;

	OptionParser opt(spikework_args);
	opt.option("-h", "--help", need_help, /* required */ false, /* as_flag */ true);
	opt.option("-v", "--verbose", verbose,  /* required */ false, /* as_flag */ true);
	opt.option("-nc", "--no-colors", no_colors,  /* required */ false, /* as_flag */ true);

	if( (args.size() == 0) || need_help) {
		cout << "Spikework usage: ./spikework [main options] [worker name] [worker options] ...\n";
		cout << "main options:\n";
		cout << "\t-h, --help        show this menu\n";
		cout << "\t-v, --verbose     verbose log level\n";
		cout << "\t-nc, --no-colors  log without colors\n";

		cout << "Available workers:\n";
		for(const auto &p: work_map) {
			cout << "\t" << p.first << "\n";
		}
		return;
	}
	opt.checkEmpty();
	if(verbose) {
		Log::inst().setLogLevel(Log::DEBUG_LEVEL);
	} else {
		Log::inst().setLogLevel(Log::INFO_LEVEL);
	}
	if(!no_colors) {
		Log::inst().setColors();
	}
	vector<string> acc_args;
	for(const auto &a: args) {
		auto p_ptr = work_map.find(a);
		if(p_ptr != work_map.end()) {
			if(workers.size() > 0) {
				workers.back()->processArgs(acc_args);
			}
			acc_args = vector<string>();
			workers.push_back(p_ptr->second());
			continue;
		}
		acc_args.push_back(a);
	}
	if(workers.size()>0) {
		workers.back()->processArgs(acc_args);
	}

	for(auto &p: workers) {
		p->start(s);
		p->process(s);
		p->end(s);
	}

}





}