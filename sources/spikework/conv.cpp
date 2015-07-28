

#include "conv.h"
#include "fft.h"

#include <dnn/util/time_series.cpp>
#include <dnn/contrib/kiss_fft/kiss_fftndr.h>
#include <dnn/util/option_parser.h>


namespace dnn {

void ConvProcessor::usage() {
    cout << "ConvProcessor perfoming convolving input signal with filter f\n";
    cout << "   --filter,  -f  filter to convolve with\n";
    cout << "\n";
    IOProcessor::usage();
}
    
void ConvProcessor::processArgs(const vector<string> &args) {
    IOProcessor::processArgs(args);
    OptionParser op(args);
    string filter_fname;
    op.option("--filter", "-f", filter_fname, false);
    if(!filter_fname.empty()) {
        ifstream ff(filter_fname);
        Stream str(ff, Stream::Binary);
        filter.set(str.readObject<TimeSeries>());
    }
}

void ConvProcessor::start(Spikework::Stack &s) {
    IOProcessor::start(s);
    if(filter.isSet()) {
        s.push(filter);
    }
}

void ConvProcessor::process(Spikework::Stack &s) {
    size_t paddingSize;
    {
        Ptr<TimeSeries> filter = s.pop().as<TimeSeries>();
        Ptr<TimeSeries> input = s.pop().as<TimeSeries>();
        
        paddingSize = filter->length();
        
        input->padRightWithZeros(paddingSize);
        filter->padRightWithZeros(input->length()-paddingSize);

        s.push(input.as<SerializableBase>());
        s.push(filter.as<SerializableBase>());
    }
    {
        Ptr<TimeSeriesComplex> input_fft;
        Ptr<TimeSeriesComplex> filter_fft;   
        {
            FFTProcessor p;
            p.process(s);
            filter_fft = s.pop().as<TimeSeriesComplex>();
            p.process(s);
            input_fft = s.pop().as<TimeSeriesComplex>();
        }        
        input_fft.ref() * filter_fft.ref();    
        s.push(input_fft);    
    }
    {
        FFTProcessor p(true); 
        p.process(s);
    }
    {
        Ptr<TimeSeries> output = s.pop().as<TimeSeries>();
        output->cutFromRight(paddingSize);
        s.push(output);
    }
}


}

