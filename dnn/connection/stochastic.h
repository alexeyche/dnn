#pragma once

#include <dnn/connection/connection.h>
#include <dnn/neuron/spike_neuron_impl.h>
#include <dnn/protos/stochastic.pb.h>

namespace NDnn {

    struct TStochasticConst : public IProtoSerial<NDnnProto::TStochasticConst> {
        static const auto ProtoFieldNumber = NDnnProto::TConnection::kStochasticFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(Prob);
            serial(InhibitoryNeuronProb);
        }

        double Prob = 1.0;
        double InhibitoryNeuronProb = 0.0;
    };


    class TStochastic : public TConnection<TStochasticConst> {
    private:
        std::unordered_map<ui32, ESourceNeuronType> SourceNeuronInfo;

    public:
        TConnectionRecipe GetConnectionRecipe(const TNeuronSpaceInfo& left, const TNeuronSpaceInfo& right) override final {
            TConnectionRecipe recipe;
        	if(c.Prob > Rand->GetUnif()) {
        		recipe.Exists = true;
        	}

            if (recipe.Exists && (c.InhibitoryNeuronProb > 0.0)) {
                auto neuronPtr = SourceNeuronInfo.find(left.GlobalId);
                if (neuronPtr == SourceNeuronInfo.end()) {
                    if (c.InhibitoryNeuronProb > Rand->GetUnif()) {
                        recipe.SourceNeuronType = SNT_INHIBITORY;
                    } else {
                        recipe.SourceNeuronType = SNT_EXCITATORY;
                    }
                    SourceNeuronInfo.emplace(left.GlobalId, recipe.SourceNeuronType);
                } else {
                    recipe.SourceNeuronType = neuronPtr->second;
                }
            }
        	return recipe;
        }
    };





} // namespace NDnn