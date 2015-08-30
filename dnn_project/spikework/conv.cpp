

#include "conv.h"
#include "fft.h"

#include <dnn/util/log/log.h>
#include <dnn/util/time_series.cpp>
#include <dnn/contrib/kiss_fft/kiss_fftndr.h>
#include <dnn/util/option_parser.h>


namespace dnn {


void ConvWorker::usage() {
    cout << "ConvWorker perfoming convolving input signal with filter f\n";
    cout << "   --filter,  -f  filter to convolve with\n";
    cout << "\n";
    IOWorker::usage();
}

void ConvWorker::processArgs(vector<string> &args) {
    IOWorker::processArgs(args);
    OptionParser op(args);
    string filter_fname;
    op.option("--filter", "-f", filter_fname, false);
    if(!filter_fname.empty()) {
        ifstream ff(filter_fname);
        Stream str(ff, Stream::Binary);
        filter.set(str.readObject<TimeSeries>());
    }
}

void ConvWorker::start(Spikework::Stack &s) {
    IOWorker::start(s);
    if(filter.isSet()) {
        s.push(filter);
    }
}

void ConvWorker::process(Spikework::Stack &s) {
    L_DEBUG << "ConvWorker, process start";
    size_t paddingSize;
    {
        L_DEBUG << "ConvWorker, padding with zeros start";
        Ptr<TimeSeries> filter = s.pop().as<TimeSeries>();
        Ptr<TimeSeries> input = s.pop().as<TimeSeries>();

        paddingSize = filter->length();

        input->padRightWithZeros(paddingSize);
        filter->padRightWithZeros(input->length()-paddingSize);
        L_DEBUG << "ConvWorker, padding with zeros for filtering with size " << paddingSize;

        size_t paddingNfft = FFTWorker::nextpow2(input->length())-input->length();
        paddingSize += paddingNfft;
        L_DEBUG << "ConvWorker, padding with next power of 2 for fast fft with size " << paddingNfft;

        input->padRightWithZeros(paddingNfft);
        filter->padRightWithZeros(paddingNfft);

        s.push(input.as<SerializableBase>());
        s.push(filter.as<SerializableBase>());
        L_DEBUG << "ConvWorker, padding with zeros end";
    }
    {
        L_DEBUG << "ConvWorker, fft start";
        Ptr<TimeSeriesComplex> input_fft;
        Ptr<TimeSeriesComplex> filter_fft;
        {
            FFTWorker p;
            L_DEBUG << "ConvWorker, fft filter start";
            p.process(s);
            L_DEBUG << "ConvWorker, fft filter end";
            filter_fft = s.pop().as<TimeSeriesComplex>();
            L_DEBUG << "ConvWorker, fft input start";
            p.process(s);
            L_DEBUG << "ConvWorker, fft input end";
            input_fft = s.pop().as<TimeSeriesComplex>();
        }
        L_DEBUG << "ConvWorker, inner product start";
        input_fft.ref() * filter_fft.ref();
        L_DEBUG << "ConvWorker, inner product end";
        s.push(input_fft);
    }
    {
        FFTWorker p(true);
        L_DEBUG << "ConvWorker, inv fft start";
        p.process(s);
        L_DEBUG << "ConvWorker, inv fft end";
    }
    {
        L_DEBUG << "ConvWorker, cut start";
        Ptr<TimeSeries> output = s.pop().as<TimeSeries>();
        output->cutFromRight(paddingSize);
        s.push(output);
        L_DEBUG << "ConvWorker, cut end";
    }
    L_DEBUG << "ConvWorker, process end";
}


}

