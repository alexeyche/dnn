#pragma once

#include <dnn/util/serial/proto_serial.h>

#include <dnn/protos/config.pb.h>

namespace NDnn {

	template <typename TConstants>
	class TReceptiveField: public IProtoSerial<NDnnProto::TLayer> {
	public:
		void SerialProcess(TProtoSerial& serial) override final {
			serial(c, TConstants::ProtoFieldNumber); 
		}
		
	protected:
		TConstants c;
	};


} // namespace NDnn


