#pragma once

#include <ground/serial/proto_serial.h>
#include <ground/ts/time_series.h>

#include <dnn/spikework/protos/spikework_config.pb.h>

#include <dnn/spikework/convolve.h>

namespace NDnn {

    class IPreprocessor: public IProtoSerial<NDnnProto::TPreprocessorConfig> {
    public:
        virtual TTimeSeries Preprocess(TTimeSeries&& ts) const = 0;

        virtual ~IPreprocessor() {}
        
    };
    
    class IFilter: public IPreprocessor {
    public:
        virtual TTimeSeries GetFilter() const = 0;

        TTimeSeries Preprocess(TTimeSeries&& ts) const override final {
            TTimeSeries filter = GetFilter();
            return Convolve(ts, filter);
        }
    };


    class IKernel: public IProtoSerial<NDnnProto::TKernelConfig> {
    public:
        virtual double PointSimilarity(const TVector<double>& x, const TVector<double>& y) const = 0;

        double Similarity(const TTimeSeries& x, const TTimeSeries& y) const {
            ENSURE((x.Dim() == y.Dim()) && (x.Length() == y.Length()), 
                "Time series must be the same shape, dimensions: " << x.Dim() << " and " << y.Dim() << ", lengths: " << x.Length() << " and " << y.Length());
            
            double integral = 0.0;
            for(ui32 i=0; i<x.Length(); ++i) {
                integral += PointSimilarity(x.GetColumnVector(i), y.GetColumnVector(i));
            }
            return integral/x.Length();
        }

        virtual ~IKernel() {}
    };

} // namespace NDnn