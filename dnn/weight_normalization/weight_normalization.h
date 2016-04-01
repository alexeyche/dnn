#pragma once

#include <dnn/util/serial/proto_serial.h>
#include <dnn/protos/config.pb.h>
#include <dnn/util/ptr.h>

namespace NDnn {


	template <typename TConstants, typename TState>
	class TWeightNormalization: public IProtoSerial<NDnnProto::TLayer> {
	public:
		void SerialProcess(TProtoSerial& serial) override final {
			serial(c, TConstants::ProtoFieldNumber);
			serial(s, TState::ProtoFieldNumber);
		}

		const TState& State() const {
			return s;
		}

		double Ltp(const double& w) const {
			return 1.0;
		}

		double Ltd(const double& w) const {
			return 1.0;
		}

		double DerivativeModulation(const double& w) const {
			return 1.0;
		}

		void CalculateDynamics(const TTime& t) {
		}

	protected:
		TConstants c;
		TState s;
	};

	template <>
	class TWeightNormalization<TEmpty, TEmpty>: public IProtoSerial<NDnnProto::TLayer> {
	public:
		void SerialProcess(TProtoSerial& serial) {
		}

		double Ltp(const double& w) const {
			return 1.0;
		}

		double Ltd(const double& w) const {
			return 1.0;
		}

		double DerivativeModulation(const double& w) const {
			return 1.0;
		}

		void CalculateDynamics(const TTime& t) {
		}
	};

	using TNoWeightNormalization = TWeightNormalization<TEmpty, TEmpty>;

} // namespace NDnn
