#pragma once

#include <dnn/neuron/defaults.h>
#include <dnn/activation/determ.h>
#include <dnn/synapse/basic_synapse.h>

namespace NDnn {

	template <
		typename TSynapseType = TBasicSynapse,
		typename TActivationFunctionType = TDeterm,
		typename TReceptiveFieldType = TNoInput,
		template <typename> class TLearningRuleType = TNoLearning,
		template <typename> class TWeightNormalizationType = TNoWeightNormalization,
		template <typename> class TIntrinsicPlasticityType = TNoIntrinsicPlasticity,
		template <typename> class TReinforcementType = TNoReinforcement
	>
	struct TNeuronConfig {
		using TNeuronSynapse = TSynapseType;
		using TNeuronActivationFunction = TActivationFunctionType;
		using TNeuronReceptiveField = TReceptiveFieldType;

		template <typename T>
		using TNeuronLearningRule = TLearningRuleType<T>;
		
		template <typename T>
		using TWeightNormalization = TWeightNormalizationType<T>;

		template <typename T>
		using TNeuronIntrinsicPlasticity = TIntrinsicPlasticityType<T>;

		template <typename T>
		using TNeuronReinforcement = TReinforcementType<T>;
	};


} // namespace NDnn
