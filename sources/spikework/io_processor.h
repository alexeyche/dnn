#pragma once

#include "processor.h"

#include <dnn/util/option_parser.h>

namespace dnn {


class IOProcessor : public Processor {
public:
	IOProcessor() : tee(false) {}
	
	void usage();
    void processArgs(const vector<string> &args);    
    void start(Spikework::Stack &s);
    void end(Spikework::Stack &s);
    
private:
    string input_filename;
    string output_filename;
    bool tee;
};


}