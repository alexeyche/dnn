

#include "fft.h"

#include <dnn/util/time_series.cpp>
#include <dnn/contrib/kiss_fft/kiss_fftndr.h>
#include <dnn/util/option_parser.h>


namespace dnn {

void FFTProcessor::usage() {
	cout << "FFTProcessor perfoming DFT on time series or spikes\n";
	cout << "	--inverse,	-inv  flag meaning inverse fft\n";
	cout << "\n";
	IOProcessor::usage();
}
	
void FFTProcessor::processArgs(const vector<string> &args) {
    IOProcessor::processArgs(args);
	OptionParser op(args);
	op.option("--inverse", "-inv", inverse, false, true);
}

void FFTProcessor::fft(const vector<double> &src, vector<complex<double>> &dst) {
	size_t vec_size = src.size();
	if(vec_size % 2 != 0) {
		cout << vec_size % 2 << "\n";
        vec_size--;
	}

	kiss_fft_cpx *data = (kiss_fft_cpx*)malloc(vec_size*sizeof(kiss_fft_cpx));
	
	for(size_t val_i=0; val_i<vec_size; ++val_i) {
    	data[val_i].r = src[val_i];
    	data[val_i].i = 0;
    }

	kiss_fft_cfg c = kiss_fft_alloc(vec_size, 0, 0, 0);
	
	kiss_fft_cpx *fout = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx)*vec_size);
	
	kiss_fft(c, data, fout);
    free(data);

    for(size_t val_i=0; val_i<vec_size; ++val_i) {
		dst.push_back(complex<double>(fout[val_i].r, fout[val_i].i));
	}

	free(fout);
	free(c);
}

void FFTProcessor::ifft(const vector<complex<double>> &src, vector<double> &dst) {
    size_t vec_size = src.size();
    if(vec_size % 2 != 0) {
        vec_size--;
    }

    kiss_fft_cpx *data = (kiss_fft_cpx*)malloc(vec_size*sizeof(kiss_fft_cpx));
    
    for(size_t val_i=0; val_i<vec_size; ++val_i) {
        data[val_i].r = src[val_i].real();
        data[val_i].i = src[val_i].imag();
    }

    kiss_fft_cfg c = kiss_fft_alloc(vec_size, 1, 0, 0);
    
    kiss_fft_cpx *fout = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx)*vec_size);
    
    kiss_fft(c, data, fout);
    free(data);

    for(size_t val_i=0; val_i<vec_size; ++val_i) {
        dst.push_back(fout[val_i].r/vec_size);
    }

    free(fout);
    free(c);
}


void FFTProcessor::process(Spikework::Stack &s) {
	Ptr<SerializableBase> input = s.pop();
	if(Ptr<TimeSeries> ts = input.as<TimeSeries>()) {
        if(inverse) {
            throw dnnException() << "For inverse transform expecting time series with complex data (TimeSeriesComplex)";
        }
		size_t dim_size = ts->data.size();
		
		Ptr<TimeSeriesComplex> out(Factory::inst().createObject<TimeSeriesComplex>());		

        for(size_t dim_i=0; dim_i<dim_size; ++dim_i) {
            fft(ts->getVector(dim_i), out->getVector(dim_i));
        }
		s.push(out.as<SerializableBase>());        
	} else 
    if(Ptr<TimeSeriesComplex> ts = input.as<TimeSeriesComplex>()) {
        if(!inverse) {
            throw dnnException() << "For fft transform expecting time series with real data (TimeSeries)";
        }
        size_t dim_size = ts->data.size();
        
        Ptr<TimeSeries> out(Factory::inst().createObject<TimeSeries>());
        
        for(size_t dim_i=0; dim_i<dim_size; ++dim_i) {
            ifft(ts->getVector(dim_i), out->getVector(dim_i));
        }
        s.push(out.as<SerializableBase>());
    } else {
        throw dnnException() << "Couldn't recognize input type\n";
    }
}


}

