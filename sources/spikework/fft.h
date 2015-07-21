#pragma once

#include "io_processor.h"

namespace dnn {
   
class FFTProcessor : public IOProcessor {
public:
	void usage();
	void processArgs(const vector<string> &args);
	
	static void fft(const vector<double> &src, vector<double> &dst, bool inverse); 
	void process(Spikework::Stack &s);

private:
	bool inverse;
};


}

