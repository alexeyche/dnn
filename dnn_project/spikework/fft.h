#pragma once

#include "io_processor.h"


namespace dnn {
   
class FFTProcessor : public IOProcessor {
public:
    FFTProcessor(bool _inverse = false) : inverse(_inverse) {}
	
    void usage();
	void processArgs(const vector<string> &args);
	
    static void fft(const vector<double> &src, vector<complex<double>> &dst); 
    static void ifft(const vector<complex<double>> &src, vector<double> &dst); 
	void process(Spikework::Stack &s);
    
private:
	bool inverse;
};


}

