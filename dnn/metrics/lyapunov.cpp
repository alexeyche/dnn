#include <cmath>

#include "lyapunov.h"

#include <ground/log/log.h>

namespace NDnn {

	NGround::TTimeSeries TLyapunov::CalculateMetrics(const NGround::TTimeSeries& input, const NGround::TTimeSeries& net) {
		NGround::TTimeSeries dst;
		NGround::ui32 ts_size = net.Length();
		double K = 1.0;

		double *values = new double[ts_size];
		memset(values, 0, ts_size*sizeof(*values));
		for(NGround::ui32 val_i = 0; val_i < ts_size - 1; ++val_i) {
		    for(NGround::ui32 di_net = 0; di_net < net.Dim(); ++di_net) {
		    	const NGround::TVector<double> &v_net = net.GetVector(di_net);
		    	for(NGround::ui32 di_input = 0; di_input < input.Dim(); ++di_input) {
		    		const NGround::TVector<double> &v_input = input.GetVector(di_input);

		    		double d_input = v_input[val_i + 1] - v_input[val_i];
		    		if (d_input != 0) {
		    			double d_net = v_net[val_i + 1] - v_net[val_i];
			        	/*L_DEBUG << "TLyapunov, calculating metrics " << di_net << "net dimension" << di_input << "input dimension";*/
			        	if (d_net != 0) {
			        		values[val_i] += log(fabs(d_net / d_input));
			        	}
			        }
		    	}
		    }
			dst.AddValue(0, K*values[val_i]);
		}

	    dst.Info = net.Info;

	    delete []values;

	    return dst;
	}

} // namespace NDnn