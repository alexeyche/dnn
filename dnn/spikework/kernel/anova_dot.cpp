#include "anova_dot.h"

namespace NDnn {

	double TAnovaDotKernel::PointSimilarity(const TVector<double>& x, const TVector<double>& y) const {
		double acc = 0.0;
	    for(ui32 i=0; i<x.size(); ++i) {
	        acc += std::exp(-Options.Sigma*(x[i] - y[i])*(x[i] - y[i]));
	    }
	    return std::pow(acc, Options.Power);
	}

} // namespace NDnn