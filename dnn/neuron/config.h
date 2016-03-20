#pragma once


namespace NDnn {

	using TNoInput = TEmpty;

	template <typename T>
	using TNoLearning = TEmpty;

	template <typename TSynapseType, typename TActivationFunctionType,  template <typename> class TLearningRuleType, typename TReceptiveFieldType = TNoInput>
	struct TNeuronConfig {
		using TNeuronSynapse = TSynapseType;
		using TNeuronActivationFunction = TActivationFunctionType;
		using TNeuronReceptiveField = TReceptiveFieldType;

		template <typename T>
		using TNeuronLearningRule = TLearningRuleType<T>;
	};


} // namespace NDnn
