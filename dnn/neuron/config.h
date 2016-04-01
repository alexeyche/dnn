#pragma once

#include <dnn/weight_normalization/weight_normalization.h>

namespace NDnn {

	using TNoInput = TEmpty;

	template <typename T1, typename T2>
	using TNoLearning = TEmpty;

	template <typename T>
	using TNoReinforcement = TEmpty;

	template <
		typename TSynapseType,
		typename TActivationFunctionType,
		typename TReceptiveFieldType = TNoInput,
		template <typename, typename> class TLearningRuleType = TNoLearning,
		typename TWeightNormalizationType = TNoWeightNormalization,
		template <typename> class TReinforcementType = TNoReinforcement
	>
	struct TNeuronConfig {
		using TNeuronSynapse = TSynapseType;
		using TNeuronActivationFunction = TActivationFunctionType;
		using TNeuronReceptiveField = TReceptiveFieldType;

		template <typename T>
		using TNeuronLearningRule = TLearningRuleType<T, TWeightNormalizationType>;

		template <typename T>
		using TNeuronReinforcement = TReinforcementType<T>;
	};


} // namespace NDnn
