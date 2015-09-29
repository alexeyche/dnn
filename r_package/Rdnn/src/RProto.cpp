
#include "RProto.h"




#include <dnn/base/factory.h>

Ptr<SerializableBase> RProto::convertBack(const Rcpp::List &list, const string &name) {
    TimeSeriesInfo ts_info;
    if( (name == "TimeSeries") || (name == "SpikesList") ) {
        if(list.containsElementNamed("ts_info")) {
            Rcpp::List ts_info_l = list["ts_info"];
            ts_info.labels_ids = Rcpp::as<vector<size_t>>(ts_info_l["labels_ids"]);
            ts_info.unique_labels = Rcpp::as<vector<string>>(ts_info_l["unique_labels"]);
            ts_info.labels_timeline = Rcpp::as<vector<size_t>>(ts_info_l["labels_timeline"]);
        }
    }
    if(name == "TimeSeries") {
        Ptr<TimeSeries> ts = Factory::inst().createObject<TimeSeries>();
        SEXP values = list["values"];
        if(Rf_isMatrix(values)) {
            Rcpp::NumericMatrix m(values);
            ts->dim_info.size = m.nrow();
            ts->data.resize(ts->dim_info.size);
            for(size_t i=0; i<m.nrow(); ++i) {
                for(size_t j=0; j<m.ncol(); ++j) {
                    ts->data[i].values.push_back(m(i,j));
                }
            }

        } else {
            ts->dim_info.size = 1;
            ts->data.resize(ts->dim_info.size);
            ts->data[0].values = Rcpp::as<vector<double>>(values);
        }

        ts->info = ts_info;
        return ts.as<SerializableBase>();
    }
    if(name == "SpikesList") {
        Rcpp::List spikes = list["values"];
        Ptr<SpikesList> sl = Factory::inst().createObject<SpikesList>();

        for(auto &sp_v: spikes) {
            SpikesSequence sp_seq;
            sp_seq.values = Rcpp::as<vector<double>>(sp_v);
            sl->seq.push_back(sp_seq);
        }
        sl->ts_info = ts_info;
        return sl.as<SerializableBase>();
    }
    if(name == "DoubleMatrix") {
        Rcpp::NumericMatrix m = list[0];
        Ptr<DoubleMatrix> r = Factory::inst().createObject<DoubleMatrix>();
        r->allocate(m.nrow(), m.ncol());
        for(size_t i=0; i<m.nrow(); ++i) {
            for(size_t j=0; j<m.ncol(); ++j) {
                r->setElement(i,j, m(i,j));
            }
        }
        return r.as<SerializableBase>();
    }
    if(name == "MatchingPursuitConfig") {
        Ptr<MatchingPursuitConfig> c = Factory::inst().createObject<MatchingPursuitConfig>();

        if(list.containsElementNamed("threshold")) c->threshold = list["threshold"];
        if(list.containsElementNamed("learn_iterations")) c->learn_iterations = list["learn_iterations"];
        if(list.containsElementNamed("jobs")) c->jobs = list["jobs"];
        if(list.containsElementNamed("learning_rate")) c->learning_rate = list["learning_rate"];
        if(list.containsElementNamed("filters_num")) c->filters_num = list["filters_num"];
        if(list.containsElementNamed("filter_size")) c->filter_size = list["filter_size"];
        if(list.containsElementNamed("learn")) c->learn = list["learn"];
        if(list.containsElementNamed("continue_learning")) c->continue_learning = list["continue_learning"];
        if(list.containsElementNamed("batch_size")) c->batch_size = list["batch_size"];
        if(list.containsElementNamed("seed")) c->seed = list["seed"];
        if(list.containsElementNamed("noise_sd")) c->noise_sd = list["noise_sd"];

        return c.as<SerializableBase>();
    }
    if(name == "FilterMatch") {
        Ptr<FilterMatch> m = Factory::inst().createObject<FilterMatch>();
        m->fi = list["fi"];
        m->t = list["t"];
        m->s = list["s"];

        return m.as<SerializableBase>();
    }
    ERR("Can't convert " << name );
}
