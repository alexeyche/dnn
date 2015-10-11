#pragma once

#include <dnn/util/fastapprox/fastlog.h>

namespace dnn {

class SRMMethods {
public:
    static inline double LLH(const SRMNeuron &n) {
		return LLH_formula(n.fired(), n.getFiringProbability());
    }

    static inline double LLH_given_Y(const SRMNeuron &n, const double &fired) {
		return LLH_formula(fired, n.getFiringProbability());
    }

    static inline double dLLH_dw(const SRMNeuron &n, const SynapseBase &syn, const double &tau_hebb) {
    	return dLLH_dw_formula(
    		n.getFiringProbability()
    	  , n.getActFunction().ifc().probDeriv(n.getMembranePotential())
    	  , n.getProbabilityModulation()
    	  , (double)n.fired()
    	  , syn.potential()
          , tau_hebb
    	);
    }

    static inline double dLLH_dw_given_Y(const SRMNeuron &n, const SynapseBase &syn, const double &fired, const double &tau_hebb) {
		return dLLH_dw_formula(
			n.getFiringProbability()
		  , n.getActFunction().ifc().probDeriv(n.getMembranePotential())
		  , n.getProbabilityModulation()
		  , (double)n.fired()
		  , syn.potential()
          , tau_hebb
		 );
    }

	static inline double LLH_formula(const double &fired, const double &p) {
		if(p<0.00001) return 0;
        return fired*log(p) + (1 - fired) * log(1-p);
	}
	static inline double dLLH_dw_formula(const double &p, const double &p_stroke, const double &M, const double &fired, const double &x, const double &tau_hebb) {
 		return (p_stroke/(p/M)) * (fired - p/(1.0+tau_hebb*p)) * x;
	}
};


}