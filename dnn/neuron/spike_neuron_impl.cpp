#include "spike_neuron_impl.h"

namespace NDnn {

	template <>
	void CallInit<TEmpty>(TEmpty&, const TNeuronSpaceInfo&) {}

	template <>
	double CallCalculateResponse<TEmpty>(TEmpty&, double) { return 0.0; }


} // namespace NDnn