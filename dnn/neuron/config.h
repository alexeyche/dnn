#pragma once

#include <dnn/weight_normalization/weight_normalization.h>

namespace NDnn {

	using TNoInput = TEmpty;

	template <typename T1, typename T2>
	using TNoLearning = TEmpty;

	template <
		typename TSynapseType, 
		typename TActivationFunctionType,  
		typename TReceptiveFieldType = TNoInput, 
		template <typename, typename> class TLearningRuleType = TNoLearning,
		typename TWeightNormalization = TNoWeightNormalization
	>
	struct TNeuronConfig {
		using TNeuronSynapse = TSynapseType;
		using TNeuronActivationFunction = TActivationFunctionType;
		using TNeuronReceptiveField = TReceptiveFieldType;

		template <typename T>
		using TNeuronLearningRule = TLearningRuleType<T, TWeightNormalization>;
	};


} // namespace NDnn
