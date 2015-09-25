
#include "RSim.h"

#include <dnn/base/factory.h>


void RSim::setTimeSeries(const Rcpp::NumericVector &v, const string &obj_name) {
    Rcpp::List tsl;
    tsl["values"] = v;
    Ptr<SerializableBase> ts = RProto::convertBack(tsl, "TimeSeries");
    auto slice = Factory::inst().getObjectsSlice(obj_name);
    for(auto it=slice.first; it != slice.second; ++it) {
        Factory::inst().getObject(it)->setAsInput(
            ts
        );
    }
    for(auto &n: neurons) {
        duration = std::max(duration, n.ref().getSimDuration());
    }
}

void RSim::setInputSpikes(const Rcpp::List &l, const string &obj_name) {
    Rcpp::List sl;
    sl["values"] = l;
    Ptr<SerializableBase> sp_l = RProto::convertBack(sl, "SpikesList");
    auto slice = Factory::inst().getObjectsSlice(obj_name);
    for(auto it=slice.first; it != slice.second; ++it) {
        Factory::inst().getObject(it)->setAsInput(
            sp_l
        );
    }
    for(auto &n: neurons) {
        duration = std::max(duration, n.ref().getSimDuration());
    }
}


Rcpp::List RSim::getStat() {
    Rcpp::List out;
    for(auto &n: neurons) {
        if(n.ref().getStat().on()) {
            stringstream ss;
            ss << n.ref().name() << "_" << n.ref().id();
            Statistics st = n.ref().getStat();
            out[ss.str()] = RProto::convertToList(&st);
        }
    }
    return out;
}

Rcpp::List RSim::getModel() {
    Rcpp::NumericMatrix w(neurons.size(), neurons.size());

    for(auto &n: neurons) {
        for(auto &syn: n.ref().getSynapses()) {
            w(syn.ref().idPre(), n.ref().id()) = syn.ref().weight();
        }
    }
    return Rcpp::List::create(
        Rcpp::Named("w") = w
    );
}
