#include "rbf_dot.h"

namespace NDnn {

	double TRbfDotKernel::PointSimilarity(const TVector<double>& x, const TVector<double>& y) const {
		double acc = 0.0;
	    // double x_norm = 0.0, y_norm = 0.0;
	    for (size_t i=0; i<x.size(); ++i) {
	        acc += Options.Sigma*(x[i] - y[i])*(x[i] - y[i]);
	        // x_norm += x[i]*x[i];
	        // y_norm += y[i]*y[i];
	    }
	    acc = std::exp(-acc);
	    // x_norm = sqrt(x_norm);
	    // y_norm = sqrt(y_norm);
	    // return acc/(x_norm*y_norm);
	    return acc;
	}

} // namespace NDnn