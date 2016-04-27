#pragma once

#include "receptive_field.h"

#include <dnn/util/serial/proto_serial.h>
#include <dnn/protos/ident.pb.h>

namespace NDnn {

    struct TIdentReceptiveFieldConst: public IProtoSerial<NDnnProto::TIdentReceptiveFieldConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kIdentReceptiveFieldFieldNumber;

        void SerialProcess(TProtoSerial& serial) override final {}
    };


    class TIdentReceptiveField : public TReceptiveField<TIdentReceptiveFieldConst> {
    public:
        void Init(const TNeuronSpaceInfo&, TRandEngine&) {
        }

        double CalculateResponse(double I) const {
            return I;
        }
    };

}
