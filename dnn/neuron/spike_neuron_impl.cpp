#include "spike_neuron_impl.h"

namespace NDnn {

	template <>
	void CallInitReceptiveField<TEmpty>(TEmpty&, const TNeuronSpaceInfo&) {}

	template <>
	double CallCalculateResponseReceptiveField<TEmpty>(TEmpty&, double) { return 0.0; }

	template <>
	void CallPropagateSynapseSpikeLearningRule<TEmpty>(TEmpty&, const TSynSpike&) {}

	template <>
	void CallCalculateDynamicsLearningRule<TEmpty>(TEmpty&, const TTime&) {}


} // namespace NDnn