#pragma once


namespace NDnn {


	template <typename TSynapseType, typename TActivationFunctionType, typename TReceptiveFieldType, bool HasInputTpl = false>
	struct TNeuronConfig {
		using TNeuronSynapse = TSynapseType;
		using TNeuronActivationFunction = TActivationFunctionType;
		using TNeuronReceptiveField = TReceptiveFieldType;

		static const bool HasInput = HasInputTpl;
	};


} // namespace NDnn
