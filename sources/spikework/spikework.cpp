
#include "spikework.h"
#include "fft.h"

#include <dnn/util/option_parser.h>
#include <dnn/io/stream.h>



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
		p->start(f);
		p->process(f);
		p->end(f);
	}

}

const char * usage = R"USAGE(Processors has options:
	--input,  -i  FILENAME specifying input of processor  (optional)
	--output, -o  FILENAME specifying output of processor (optional)
	--help,   -h  show this help message
)USAGE";

void Processor::processDefaultArgs(const vector<string> &args) {
	OptionParser op(args);
    bool need_help = false;
	op.option("--input", "-i", input_filename, false);
	op.option("--output", "-o", output_filename, false);
    op.option("--help", "-h", need_help, false, true);
    processArgs(args);
    if(need_help) {
    	cout << usage;
    	std::exit(0);
    }
}

void Processor::start(Spikework::Field &f) {
	if(!input_filename.empty()) {
		ifstream ff(input_filename);
	    Stream s(ff, Stream::Binary);
	    f.push_input(Ptr<SerializableBase>(s.readBaseObject()));
	}
}



}