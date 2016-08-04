#include "fft.h"

#include <dnn/contrib/kiss_fft/kiss_fftndr.h>
#include <ground/log/log.h>

namespace NDnn {

	TTimeSeriesComplex TFFT::Transform(const TTimeSeries& src) {
		TTimeSeriesComplex dst;

		ui32 ts_size = src.Length();
	    if (ts_size % 2 != 0) {
	        ts_size--;
	    }
	    L_DEBUG << "TFFT, fft start, nfft == " << ts_size;
	    kiss_fftr_cfg c = kiss_fftr_alloc(ts_size, false, 0, 0);

	    kiss_fft_scalar *data_in = new kiss_fft_scalar[ts_size];
	    ui32 out_ts_size = ts_size/2 + 1;
	    kiss_fft_cpx *data_out = new kiss_fft_cpx[out_ts_size];

	    for(ui32 di=0; di<src.Dim(); ++di) {
	        L_DEBUG << "TFFT, fft processing " << di << " dimension";
	        const TVector<double> &v = src.GetVector(di);
	        for(ui32 val_i=0; val_i<ts_size; ++val_i) {
	            data_in[val_i] = v[val_i];
	        }
	        kiss_fftr(c, data_in, data_out);

	        for(ui32 val_i=0; val_i<out_ts_size; ++val_i) {
	            dst.AddValue(
	                di,
	                TComplex(
	                    data_out[val_i].r,
	                    data_out[val_i].i
	                )
	            );
	        }
	    }
	    dst.Info = src.Info;
	    L_DEBUG << "TFFT, fft end, out size == " << out_ts_size;

	    delete []data_out;
	    delete []data_in;
	    free(c);

	    return dst;
	}

	TTimeSeries TFFT::TransformBack(const TTimeSeriesComplex& src) {
		TTimeSeries dst;
		ui32 ts_size = src.Length();

	    ui32 out_size = (ts_size-1)*2;

	    kiss_fftr_cfg c = kiss_fftr_alloc(out_size, 1, 0, 0);

	    kiss_fft_cpx *data_in = new kiss_fft_cpx[ts_size];
	    kiss_fft_scalar *data_out = new kiss_fft_scalar[out_size];

	    for(ui32 di=0; di<src.Dim(); ++di) {
	        L_DEBUG << "TFFT, fft inverse processing " << di << " dimension";
	        const TVector<TComplex>& v = src.GetVector(di);
	        for(ui32 val_i=0; val_i<ts_size; ++val_i) {
	            data_in[val_i].r = v[val_i].real();
	            data_in[val_i].i = v[val_i].imag();
	        }
	        kiss_fftri(c, data_in, data_out);

	        for(ui32 val_i=0; val_i<out_size; ++val_i) {
	            dst.AddValue(di, data_out[val_i]/out_size);
	        }
	    }
	    dst.Info = src.Info;

	    delete []data_out;
	    delete []data_in;
	    free(c);

	    return dst;
	}

	ui32 TFFT::Nextpow2(ui32 s) {
		return kiss_fftr_next_fast_size_real(s);
	}

} // namespace NDnn