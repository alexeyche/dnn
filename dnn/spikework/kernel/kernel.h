#pragma once

#include <dnn/util/serial/proto_serial.h>
#include <dnn/util/ts/time_series.h>

#include <dnn/spikework/protos/kernel_options.pb.h>

#include <dnn/spikework/convolve.h>

namespace NDnn {

    template <typename TOptions>
    class IPreprocessor: public IProtoSerial<NDnnProto::TKernelOptions> {
    public:
        void SerialProcess(TProtoSerial& serial) {
            serial(Options, TOptions::ProtoFieldNumber);
        }

        virtual TTimeSeries Preprocess(TTimeSeries&& ts) const = 0;

        virtual ~IPreprocessor() {}

    protected:
        TOptions Options;
    };


    template <typename TOptions>
    class IFilter: public IPreprocessor<TOptions> {
    public:
        virtual TTimeSeries GetFilter() const = 0;

        TTimeSeries Preprocess(TTimeSeries&& ts) const override final {
            TTimeSeries filter = GetFilter();
            return Convolve(ts, filter);
        }
    };


    template <typename TOptions>
    class IKernel: public IProtoSerial<NDnnProto::TKernelOptions> {
    public:
        void SerialProcess(TProtoSerial& serial) {
            serial(Options, TOptions::ProtoFieldNumber);
        }

        virtual double Calculate(const TTimeSeries& x, const TTimeSeries& y) const = 0;

        virtual ~IKernel() {}

    protected:
        TOptions Options;
    };

} // namespace NDnn