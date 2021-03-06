#include "builder.h"


#include <dnn/connection/stochastic.h>
#include <dnn/connection/difference_of_gaussians.h>
#include <dnn/connection/random_neuron.h>

namespace NDnn {

	TConnectionPtr BuildConnection(const NDnnProto::TConnection& conn, TRandEngine& rand) {
		TConnectionPtr out;
		if (conn.has_stochastic()) {
			ENSURE(!out, "Got duplicates of connection type in connection specification: " << conn.DebugString());
			out = MakeShared(new TStochastic());
		} else
		if (conn.has_differenceofgaussians()) {
			ENSURE(!out, "Got duplicates of connection type in connection specification: " << conn.DebugString());
			out = MakeShared(new TDifferenceOfGaussians());
		} else
		if (conn.has_randomneuron()) {
			ENSURE(!out, "Got duplicates of connection type in connection specification: " << conn.DebugString());
			out = MakeShared(new TRandomNeuron());
		}

		ENSURE(out, "Connection is not implemented for " << conn.DebugString());
		out->Deserialize(conn);
		out->SetRandEngine(rand);
		return out;
	}


} // namespace NDnn