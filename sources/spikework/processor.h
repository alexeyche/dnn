#pragma once


#include "spikework.h"

namespace dnn {

class Processor {
public:
	virtual void usage() = 0;
	
	virtual void processDefaultArgs(const vector<string> &args) {};
	virtual void processArgs(const vector<string> &args) {};

	virtual void start(Spikework::Stack &s) {}
	virtual void process(Spikework::Stack &s) = 0;
	virtual void end(Spikework::Stack &s) {}
};

}