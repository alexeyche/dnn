#pragma once


#include "spikework.h"

namespace dnn {

class Worker {
public:
    virtual ~Worker() {}
	virtual void usage() = 0;

	virtual void processArgs(vector<string> &args) {};

	virtual void start(Spikework::Stack &s) {}
	virtual void process(Spikework::Stack &s) = 0;
	virtual void end(Spikework::Stack &s) {}
};

}