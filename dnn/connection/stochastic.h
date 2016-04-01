#pragma once

#include <dnn/connection/connection.h>
#include <dnn/protos/stochastic.pb.h>

namespace NDnn {

    struct TStochasticConst : public IProtoSerial<NDnnProto::TStochasticConst> {
        static const auto ProtoFieldNumber = NDnnProto::TConnection::kStochasticFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(Prob);
        }

        double Prob = 1.0;
    };


    class TStochastic : public TConnection<TStochasticConst> {
    public:
        TConnectionRecipe GetConnectionRecipe(const TNeuronSpaceInfo& left, const TNeuronSpaceInfo& right) override final {
        	TConnectionRecipe recipe;
        	if(c.Prob > Rand->GetUnif()) {
        		recipe.Exists = true;
        	}
        	return recipe;
        }
    };





} // namespace NDnn