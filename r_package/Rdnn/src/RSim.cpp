
#include "RSim.h"

#include <dnn/base/factory.h>


RSim::RSim() {
    Factory::inst().cleanHeap();
}

void RSim::setTimeSeries(SEXP v, const string &obj_name) {
    Ptr<SerializableBase> ts;
    if(Rf_isMatrix(v)) {
        Rcpp::List tsl;
        tsl["values"] = v;
        ts = RProto::convertBack(tsl, "TimeSeries");
    } else {
        try {
            ts = RProto::convertBack(v, "TimeSeries");
        } catch(...) {
            ERR("Expecting matrix with values or TimeSeries list object\n");
        }
    }

    auto slice = Factory::inst().getObjectsSlice(obj_name);
    for(auto it=slice.first; it != slice.second; ++it) {
        Factory::inst().getObject(it)->setAsInput(
            ts
        );
    }
    net->spikesList().ts_info = ts.as<TimeSeries>()->info;
}

void RSim::setInputSpikes(const Rcpp::List &l, const string &obj_name) {
    Ptr<SerializableBase> sp_l;
    if(l.containsElementNamed("values")) {
        sp_l = RProto::convertBack(l, "SpikesList");
    } else {
        try {
            Rcpp::List sl;
            sl["values"] = l;
            sp_l = RProto::convertBack(sl, "SpikesList");
        } catch (...) {
            ERR("Expecting list with spike times of neurons or SpikesList list object\n");
        }
    }

    auto slice = Factory::inst().getObjectsSlice(obj_name);
    for(auto it=slice.first; it != slice.second; ++it) {
        Factory::inst().getObject(it)->setAsInput(
            sp_l
        );
    }
    net->spikesList().ts_info = sp_l.as<SpikesList>()->ts_info;
}


Rcpp::List RSim::getStat() {
    Rcpp::List out;
    for(auto &n: neurons) {
        if(n.ref().getStat().on()) {
            stringstream ss;
            ss << n.ref().name() << "_" << n.ref().id();
            Statistics st = n.ref().getStat();
            out[ss.str()] = RProto::convertToR(&st);
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
