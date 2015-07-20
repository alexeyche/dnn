

#include "fft.h"

#include <dnn/util/time_series.cpp>

namespace dnn {


void FFTProcessor::process(Spikework::Field &f) {
	Ptr<SerializableBase> input = f.pop_input();
	if(Ptr<TimeSeries> ts = input.as<TimeSeries>()) {
		cout << "Got time series\n";
	}
}


}

