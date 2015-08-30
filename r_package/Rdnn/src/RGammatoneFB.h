#ifndef RGAMMATONEFB_H
#define RGAMMATONEFB_H

#define STRICT_R_HEADERS
#include <Rcpp.h>

#include "common.h"

#include <dnn/util/gammatone_fb.h>


class RGammatoneFB {
public:
    RGammatoneFB() {}
    Rcpp::List calc(Rcpp::NumericVector x, Rcpp::NumericVector freqs, double sampling_rate, int hrect, int verbose) {
        Rcpp::NumericMatrix membrane(freqs.size(), x.size());
        vector<vector<double>> hilbert_envelope;
        vector<vector<double>> inst_phase;
        vector<vector<double>> inst_freq;

        vector<double> xv = Rcpp::as<vector<double>>(x);
        size_t i=0;
        for(auto it = freqs.begin(); it != freqs.end(); ++it) {
            vector<double> out_f;
            GammatoneFilter f( (verbose>0) ? GammatoneFilter::Options::Full : GammatoneFilter::Options::OnlyMembrane);

            f.calc(xv, sampling_rate, *it, hrect);
            for(size_t j=0; j<f.membrane.size(); ++j) {
                membrane(i,j) = f.membrane[j];
            }

            if(verbose>0) {
                hilbert_envelope.push_back(f.hilbert_envelope);
                inst_phase.push_back(f.inst_phase);
                inst_freq.push_back(f.inst_freq);
            }
            ++i;
        }
        Rcpp::List list_out;
        list_out["membrane"] = membrane;
        if(verbose>0) {
            list_out["hilbert_envelope"] = hilbert_envelope;
            list_out["inst_phase"] = inst_phase;
            list_out["inst_freq"] = inst_freq;
        }
        return list_out;
    }
    void print() {
        cout << "GammatoneFB instance\n";
    }

    size_t verbose;
};

#endif
