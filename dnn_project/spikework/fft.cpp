

#include "fft.h"

#include <dnn/util/time_series.cpp>
#include <dnn/contrib/kiss_fft/kiss_fftndr.h>
#include <dnn/util/option_parser.h>
#include <dnn/util/log/log.h>

#include <string.h>

namespace dnn {

void FFTWorker::usage() {
	cout << "FFTWorker perfoming DFT on time series or spikes\n";
	cout << "	--inverse,	-inv  flag meaning inverse fft\n";
	cout << "\n";
	IOWorker::usage();
}

void FFTWorker::processArgs(vector<string> &args) {
    IOWorker::processArgs(args);
	OptionParser op(args);
	op.option("--inverse", "-inv", inverse, false, true);
}

void FFTWorker::fft(const TimeSeries &src, TimeSeriesComplex &dst) {
    size_t ts_size = src.length();
    if(ts_size % 2 != 0) {
        ts_size--;
    }
    L_DEBUG << "FFTWorker, fft start, nfft == " << ts_size;
    kiss_fftr_cfg c = kiss_fftr_alloc(ts_size, false, 0, 0);

    kiss_fft_scalar *data_in = new kiss_fft_scalar[ts_size];
    size_t out_ts_size = ts_size/2 + 1;
    kiss_fft_cpx *data_out = new kiss_fft_cpx[out_ts_size];

    for(size_t di=0; di<src.dim(); ++di) {
        L_DEBUG << "FFTWorker, fft processing " << di << " dimension";
        const vector<double> &v = src.getVector(di);
        for(size_t val_i=0; val_i<ts_size; ++val_i) {
            data_in[val_i] = v[val_i];
        }
        kiss_fftr(c, data_in, data_out);

        for(size_t val_i=0; val_i<out_ts_size; ++val_i) {
            dst.addValue(
                di,
                complex<double>(
                    data_out[val_i].r,
                    data_out[val_i].i
                )
            );
        }
    }
    dst.info = src.info;
    L_DEBUG << "FFTWorker, fft end, out size == " << out_ts_size;

    delete []data_out;
    delete []data_in;
    free(c);
}

void FFTWorker::ifft(const TimeSeriesComplex &src, TimeSeries &dst) {
    size_t ts_size = src.length();

    size_t out_size = (ts_size-1)*2;

    kiss_fftr_cfg c = kiss_fftr_alloc(out_size, 1, 0, 0);

    kiss_fft_cpx *data_in = new kiss_fft_cpx[ts_size];
    kiss_fft_scalar *data_out = new kiss_fft_scalar[out_size];

    for(size_t di=0; di<src.dim(); ++di) {
        L_DEBUG << "fft: inverse processing " << di << " dimension";
        const vector<complex<double>> &v = src.getVector(di);
        for(size_t val_i=0; val_i<ts_size; ++val_i) {
            data_in[val_i].r = v[val_i].real();
            data_in[val_i].i = v[val_i].imag();
        }
        kiss_fftri(c, data_in, data_out);

        for(size_t val_i=0; val_i<out_size; ++val_i) {
            dst.addValue(di, data_out[val_i]/out_size);
        }
    }
    dst.info = src.info;

    delete []data_out;
    delete []data_in;
    free(c);
}


void FFTWorker::process(Spikework::Stack &s) {
	Ptr<SerializableBase> input = s.pop();
	if(Ptr<TimeSeries> ts = input.as<TimeSeries>()) {
        L_DEBUG << "FFTWorker, start";
        if(inverse) {
            throw dnnException() << "For inverse transform expecting time series with complex data (TimeSeriesComplex)";
        }
		size_t dim_size = ts->data.size();

		Ptr<TimeSeriesComplex> out(Factory::inst().createObject<TimeSeriesComplex>());

        fft(ts.ref(), out.ref());
		s.push(out.as<SerializableBase>());
        L_DEBUG << "FFTWorker, end";
	} else
    if(Ptr<TimeSeriesComplex> ts = input.as<TimeSeriesComplex>()) {
        L_DEBUG << "FFTWorker, inverse, start";
        if(!inverse) {
            throw dnnException() << "For fft transform expecting time series with real data (TimeSeries)";
        }

        Ptr<TimeSeries> out(Factory::inst().createObject<TimeSeries>());
        ifft(ts.ref(), out.ref());
        s.push(out.as<SerializableBase>());
        L_DEBUG << "FFTWorker, inverse, end";
    } else {
        throw dnnException() << "Couldn't recognize input type\n";
    }
}

size_t FFTWorker::nextpow2(const size_t &s) {
    return kiss_fft_next_fast_size(s);
}

}

