
#include "spikework.h"
#include "fft.h"
#include "conv.h"
#include "kernel.h"
#include "gram.h"
#include "read.h"

namespace dnn {


Spikework::~Spikework() {
    for(auto &v: processors) {
        delete v.ptr();
    }
}

Spikework::Spikework(const vector<string> &args) {
	proc_map["read"] = &createProcessor<ReadProcessor>;
	proc_map["fft"] = &createProcessor<FFTProcessor>;
	proc_map["conv"] = &createProcessor<ConvProcessor>;
	proc_map["kernel"] = &createProcessor<KernelProcessor>;
	proc_map["gram"] = &createProcessor<GramProcessor>;
	if( (args.size() == 0) || ((args.size() == 1)&&((args[0] == "-h")||(args[0] == "--help")))) {
		cout << "Available processors:\n";
		for(const auto &p: proc_map) {
			cout << "\t" << p.first << "\n";
		}
	}

	vector<string> acc_args;
	for(const auto &a: args) {
		auto p_ptr = proc_map.find(a);
		if(p_ptr != proc_map.end()) {
			if(processors.size() > 0) {
				processors.back()->processArgs(acc_args);
			}
			acc_args = vector<string>();
			processors.push_back(p_ptr->second());
			continue;
		}
		acc_args.push_back(a);
	}
	if(processors.size()>0) {
		processors.back()->processArgs(acc_args);
	}

	for(auto &p: processors) {
		p->start(s);
		p->process(s);
		p->end(s);
	}

}





}