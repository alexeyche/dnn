#pragma once

#include "io_worker.h"


namespace dnn {

class FFTWorker : public IOWorker {
public:
    FFTWorker(bool _inverse = false) : inverse(_inverse) {}

    void usage();
	void processArgs(vector<string> &args);

    static void fft(const TimeSeries &src, TimeSeriesComplex &dst);
    static void ifft(const TimeSeriesComplex &src, TimeSeries &dst);
	void process(Spikework::Stack &s);
    static size_t nextpow2(const size_t &s);
private:
	bool inverse;
};


}

