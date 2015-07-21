
#include "spikework.h"
#include "fft.h"




namespace dnn {

Spikework::Spikework(const vector<string> &args) { 
	proc_map["fft"] = &createProcessor<FFTProcessor>;

	vector<string> acc_args;
	for(const auto &a: args) {
		auto p_ptr = proc_map.find(a);
		if(p_ptr != proc_map.end()) {
			if(processors.size() > 0) {
				processors.back()->processDefaultArgs(acc_args);				
			}
			acc_args = vector<string>();
			processors.push_back(p_ptr->second());
			continue;
		}
		acc_args.push_back(a);
	}
	if(processors.size()>0) {
		processors.back()->processDefaultArgs(acc_args);
	}

	for(auto &p: processors) {
		p->start(s);
		p->process(s);
		p->end(s);
	}

}





}