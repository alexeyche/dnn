
#include "io_processor.h"

#include <dnn/io/stream.h>

namespace dnn {

void IOProcessor::usage() {
	cout << "Processors has options:\n";
	cout << "	--input,  -i  FILENAME specifying input of processor  (optional)\n";
	cout << "	--output, -o  FILENAME specifying output of processor (optional)\n";
	cout << "	--tee,    -t  flag meaning to dump output to file without deleting from stack (optional)\n";
	cout << "	--help,   -h  show this help message\n";
}


void IOProcessor::processArgs(const vector<string> &args) {
	OptionParser op(args);
    bool need_help = false;
	op.option("--input", "-i", input_filename, false);
	op.option("--output", "-o", output_filename, false);
    op.option("--help", "-h", need_help, false, true);
    if(need_help) {
    	usage();
    	std::exit(0);
    }
}

void IOProcessor::start(Spikework::Stack &s) {
	if(!input_filename.empty()) {
		ifstream ff(input_filename);
	    Stream str(ff, Stream::Binary);
        Ptr<SerializableBase> o = str.readBaseObject();
        // if(o.as<SpikesList>()) {
        //     throw dnnException() << "Not implemented\n";
        // }
	    s.push(o);
	}
}

void IOProcessor::end(Spikework::Stack &s) {
	if(!output_filename.empty()) {
		Ptr<SerializableBase> p;
		if(tee) {
			p = s.back();
		} else {
			p = s.pop();
		}

		ofstream ff(output_filename);
	    Stream str(ff, Stream::Binary);
        str.writeObject(p.ptr());
	}
}


}