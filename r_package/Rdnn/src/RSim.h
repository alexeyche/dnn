#ifndef RSIM_H
#define RSIM_H


#include <dnn/sim/sim.h>

#undef PI
#define STRICT_R_HEADERS
#include <Rcpp.h>

#include "RConstants.h"
#include "RProto.h"

class RSim : public dnn::Sim {
public:
    RSim() {

    }
    // RSim(RConstants *rc) try : Sim(*rc)  {
    //     global_neuron_index = 0;
    //     Sim::build();
    // } catch (std::exception &e) {
    //     ERR("\nCan't build Sim: " << e.what() << "\n");
    // }
    ~RSim() { }

    void print() {
        cout << *this;
    }
    void build() {
        try {
            Sim::build();
        } catch (std::exception &e) {
            ERR("\nCan't build Sim: " << e.what() << "\n");
        }
    }

    RConstants getConst() {
        return RConstants(Sim::c);
    }

    void run(size_t jobs=1) {
        try {
            Sim::run(jobs);
        } catch(std::exception &e) {
            ERR("Sim run failed: " << e.what() << "\n" );
        }
    }

    void setTimeSeries(const Rcpp::NumericVector &v, const string &obj_name);

    void setInputSpikes(const Rcpp::List &l, const string &obj_name);

    Rcpp::List getStat();

    Rcpp::List getModel();

    void turnOnStatistics() {
        Sim::turnOnStatistics();
    }
    void saveModel(const string &fname) {
        ofstream fstr(fname);
        Stream str_out(fstr, Stream::Binary);
        serialize(str_out);
    }
    Rcpp::List getSpikes() {
        if(!net.get()) {
            throw dnnException()<< "Sim network was not found. You need to build sim\n";
        }
        return RProto::convertToList(&net->spikesList());
    }
};





#endif
