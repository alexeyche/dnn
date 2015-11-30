#pragma once

#include <dnn/connections/connection.h>
#include <dnn/protos/stochastic.pb.h>

namespace dnn {

/*@GENERATE_PROTO@*/
struct StochasticC : public Serializable<Protos::StochasticC> {
    StochasticC() : prob(0.0) {}

    void serial_process() {
        begin() << "prob: " << prob << Self::end;
    }

    double prob;
};


class Stochastic : public Connection<StochasticC> {
public:
    ConnectionRecipe getConnectionRecipe(const SpikeNeuronBase &left, const SpikeNeuronBase &right) {
    	ConnectionRecipe recipe;
    	if(c.prob > getUnif()) {
    		recipe.exists = true;
    	} else {
    		recipe.exists = false;
    	}
    	return recipe;
    }
};





};