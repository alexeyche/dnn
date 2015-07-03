#pragma once

#include <dnn/protos/generated.pb.h>
#include <dnn/io/serialize.h>

namespace dnn {

/*@GENERATE_PROTO@*/
struct SimInfo : public Serializable<Protos::SimInfo>  {
	SimInfo() : pastTime(0.0) {}

	void serial_process() {
		begin() << "pastTime: " << pastTime << Self::end;
	}

	double pastTime;
};

}