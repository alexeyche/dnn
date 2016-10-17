#pragma once

#include <dnn/connection/connection.h>
#include <dnn/protos/stochastic.pb.h>

namespace NDnn {

    struct TRandomNeuronConst : public IProtoSerial<NDnnProto::TRandomNeuronConst> {
        static const auto ProtoFieldNumber = NDnnProto::TConnection::kRandomNeuronFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(NeuronProb);
            serial(SynapseProb);
        }

        double NeuronProb = 1.0;
        double SynapseProb = 1.0;
    };


    class TRandomNeuron : public TConnection<TRandomNeuronConst> {
    public:
        TConnectionRecipe GetConnectionRecipe(const TNeuronSpaceInfo& left, const TNeuronSpaceInfo& right) override final {
        	TConnectionRecipe recipe;

            auto decisionPtr = NeuronDecisions.find(right.GlobalId);
            if (decisionPtr == NeuronDecisions.end()) {
                auto res = NeuronDecisions.emplace(right.GlobalId, c.NeuronProb > Rand->GetUnif());
                decisionPtr = res.first;
            }

            if (decisionPtr->second) {
                recipe.Exists = c.SynapseProb > Rand->GetUnif();
            }

            return recipe;
        }

        TMap<ui32, bool> NeuronDecisions;
    };





} // namespace NDnn