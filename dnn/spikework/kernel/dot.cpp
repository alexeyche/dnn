#include "dot.h"

namespace NDnn {

	double TDotKernel::PointSimilarity(const TVector<double>& x, const TVector<double>& y) const {
		double acc = 0.0;
	    double x_norm = 0.0, y_norm = 0.0;
	    for(ui32 i=0; i<x.size(); ++i) {
	        acc += x[i]*y[i];
	        x_norm += x[i]*x[i];
	        y_norm += y[i]*y[i];
	    }
	    x_norm = sqrt(x_norm);
	    y_norm = sqrt(y_norm);
	    if ((std::fabs(x_norm) < std::numeric_limits<double>::epsilon()) || (std::fabs(y_norm) < std::numeric_limits<double>::epsilon())) {
	    	return 0.0;
	    }
	    return acc/(x_norm*y_norm);
	}

} // namespace NDnn