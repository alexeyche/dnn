#pragma once


namespace NDnn {

	using TNoInput = TEmpty;

	template <typename TSynapseType, typename TActivationFunctionType, typename TReceptiveFieldType = TNoInput>
	struct TNeuronConfig {
		using TNeuronSynapse = TSynapseType;
		using TNeuronActivationFunction = TActivationFunctionType;
		using TNeuronReceptiveField = TReceptiveFieldType;
	};


} // namespace NDnn
