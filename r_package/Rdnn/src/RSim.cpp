
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
        tsl.attr("class") = "TimeSeries";
        ts = RProto::convertFromR<SerializableBase>(tsl);
    } else {
        try {
            ts = RProto::convertFromR<SerializableBase>(v);
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
    net->spikesList().info = ts.as<TimeSeries>()->info;
    for(auto &n: neurons) {
        n.ref().initInternal();
    }
}

void RSim::setInputSpikes(const Rcpp::List &l, const string &obj_name) {
    Ptr<SerializableBase> sp_l;
    if(l.containsElementNamed("values")) {
        sp_l = RProto::convertFromR<SerializableBase>(l);
    } else {
        try {
            Rcpp::List sl;
            sl["values"] = l;
            sl.attr("class") = "SpikesList";
            sp_l = RProto::convertFromR<SerializableBase>(sl);
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
    net->spikesList().info = sp_l.as<SpikesList>()->info;
    for(auto &n: neurons) {
        n.ref().initInternal();
    }
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
