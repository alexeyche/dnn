#include <cmath>

#include "lyapunov.h"

#include <dnn/util/log/log.h>

namespace NDnn {

	TTimeSeries TLyapunov::CalculateMetrics(const TTimeSeries& input, const TTimeSeries& net) {
		TTimeSeries dst;
		ui32 ts_size = net.Length();
		double K = 1.0

		double *values = new double[ts_size];
		memset(values, 0, ts_size*sizeof(*values))
		for(ui32 val_i = 0; val_i < ts_size - 1; ++val_i) {
		    for(ui32 di_net = 0; di_net < net.Dim(); ++di_net) {
		    	const TVector<TComplex>& v_net = net.GetVector(di_net);
		    	for(ui32 di_input = 0; di_input < input.Dim(); ++di_input) {
		    		const TVector<TComplex>& v_input = input.GetVector(di_input);

		    		double d_input = v_input[val_i + 1].real() - v_input[val_i].real();
		    		if d_input:
		    			double d_net = v_net[val_i + 1].real() - v_net[val_i].real();
			        	/*L_DEBUG << "TLyapunov, calculating metrics " << di_net << "net dimension" << di_input << "input dimension";*/
			        	values[val_i] += log(d_net / d_input);
		    	}
		    }
			dst.AddValue(0, K*values[val_i]);
		}

	    dst.Info = net.Info;

	    delete []values;

	    return dst;
	}

} // namespace NDnn