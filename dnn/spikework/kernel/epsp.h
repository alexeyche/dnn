#pragma once

#include "kernel.h"

#include <dnn/spikework/protos/spikework_config.pb.h>

namespace NDnn {

    struct TEpspOptions: IProtoSerial<NDnnProto::TEpspOptions> {
        void SerialProcess(TProtoSerial& serial) {
            serial(TauRise);
            serial(TauDecay);
            serial(Length);
            serial(Dt);
        }

        double TauRise = 0.0;
        double TauDecay = 50.0;
        double Length = 100.0;
        double Dt = 1.0;
    };


    class TEpspFilter: public IFilter {
    public:
        TTimeSeries GetFilter() const override final {
            TEpspOptions opts = Options;

            TFunction<double(double)> fun;
            if(opts.TauRise>0.0001) {
                if(fabs(opts.TauRise - opts.TauDecay) < 1e-06) {
                    opts.TauDecay += 1e-05;
                }
                fun = [=](double t) {
                    return (1.0/(1.0-opts.TauRise/opts.TauDecay)) * (exp(-t/opts.TauDecay) - exp(-t/opts.TauRise));
                };
            } else {
                fun = [=](double t) {
                    return exp(-t/opts.TauDecay);
                };
            }
            TTimeSeries filter;
            for(size_t di=0; di<1; ++di) { // 1 because TimeSeries can deal with 1 dimensional while inner product
                double max_t = opts.Length * opts.Dt;
                for(double s=0; s<max_t; s+=opts.Dt) {
                    filter.AddValue(di, fun(s));
                }
            }
            return filter;
        }

        void SerialProcess(TProtoSerial& serial) override {
            serial(Options, NDnnProto::TPreprocessorConfig::kEpspFieldNumber);
        }

    private:
        TEpspOptions Options;
    };



} // namespace NDnn