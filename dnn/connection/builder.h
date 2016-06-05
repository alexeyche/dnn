#pragma once

#include <dnn/connection/connection.h>
#include <dnn/protos/config.pb.h>
#include <ground/rand.h>

namespace NDnn {

	TConnectionPtr BuildConnection(const NDnnProto::TConnection& conn, TRandEngine& rand);


} // namespace NDnn