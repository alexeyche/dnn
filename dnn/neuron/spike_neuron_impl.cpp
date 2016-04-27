#include "spike_neuron_impl.h"

namespace NDnn {

	template <>
	void CallInitReceptiveField<TEmpty>(TEmpty&,const TNeuronSpaceInfo&, TRandEngine& rand) {}

	template <>
	double CallCalculateResponseReceptiveField<TEmpty>(TEmpty&, double) { return 0.0; }

	template <>
	void CallModulateRewardReinforcement<TEmpty>(TEmpty&) {}


} // namespace NDnn