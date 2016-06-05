#pragma once

#include "kernel.h"

#include <dnn/spikework/protos/spikework_config.pb.h>

namespace NDnn {

    struct TGaussOptions: IProtoSerial<NDnnProto::TGaussOptions> {
        void SerialProcess(TProtoSerial& serial) {
            serial(Sigma);
            serial(Length);
            serial(Dt);
        }

        double Sigma = 0.1;
        double Length = 100.0;
        double Dt = 1.0;
    };


    class TGaussFilter: public IFilter {
    public:
        TTimeSeries GetFilter() const override final {
            TGaussOptions opts = Options;

            TTimeSeries filter;
            for(size_t di=0; di<1; ++di) { // 1 because TimeSeries can deal with 1 dimensional while inner product
                double max_t = opts.Length * opts.Dt;
                for(double s=0; s<max_t; s+=opts.Dt) {
                    filter.AddValue(
                        di, 
                        std::exp( - (s - opts.Length/2.0) * (s - opts.Length/2.0) / (2.0 * opts.Sigma * opts.Sigma) )
                    );
                }
            }
            return filter;
        }

        void SerialProcess(TProtoSerial& serial) override {
            serial(Options, NDnnProto::TPreprocessorConfig::kGaussFieldNumber);
        }

    private:
        TGaussOptions Options;
    };



} // namespace NDnn