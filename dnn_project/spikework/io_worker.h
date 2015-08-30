#pragma once

#include "worker.h"

#include <dnn/util/option_parser.h>

namespace dnn {


class IOWorker : public Worker {
public:
	IOWorker() : tee(false), dt(1.0) {}

	void usage();
    void processArgs(vector<string> &args);
    void start(Spikework::Stack &s);
    void end(Spikework::Stack &s);

private:
    string input_filename;
    string output_filename;
    bool tee;
    double dt;
};


}