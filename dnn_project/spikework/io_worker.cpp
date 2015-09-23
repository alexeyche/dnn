
#include "io_worker.h"

#include <dnn/io/stream.h>
#include <dnn/util/spikes_list.h>

namespace dnn {

void IOWorker::usage() {
	cout << "Workers has options:\n";
	cout << "	--input,  -i  FILENAME specifying input of worker  (optional)\n";
	cout << "	--output, -o  FILENAME specifying output of worker (optional)\n";
	cout << "	--tee,    -t  flag meaning to dump output to file without deleting from stack (optional)\n";
    cout << "        --dt,         spike lists converted into time series with specified resolution (default: " << dt << ")\n";
	cout << "	--help,   -h  show this help message\n";
}


void IOWorker::processArgs(vector<string> &args) {
	OptionParser op(args);
    bool need_help = false;
	op.option("--input", "-i", input_filename, false);
	op.option("--output", "-o", output_filename, false);
    op.option("--help", "-h", need_help, false, true);
    op.loption("--dt", dt, false);
    if(need_help) {
    	usage();
    	std::exit(0);
    }
    args = op.getRawOptions();
}

void IOWorker::start(Spikework::Stack &s) {
	if(!input_filename.empty()) {
		ifstream ff(input_filename);
	    Stream str(ff, Stream::Binary);
        Ptr<SerializableBase> o = str.readBase();
        if(Ptr<SpikesList> sp = o.as<SpikesList>()) {
            s.push(sp->convertToBinaryTimeSeries(dt));
        } else {
            s.push(o);
        }
	}
}

void IOWorker::end(Spikework::Stack &s) {
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