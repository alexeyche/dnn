#ifndef RPROTO_READ_H
#define RPROTO_READ_H

#include <dnn/io/stream.h>
#include <dnn/util/matrix.h>
#include <mpl/mpl.h>
#include <shapelets/subsequence.h>
#include <dnn/util/statistics.h>
#include <dnn/neurons/spike_neuron.h>


#undef PI
#define STRICT_R_HEADERS
#include <Rcpp.h>

// #include <R.h>
// #include <Rinternals.h>


#include "common.h"

using namespace dnn;
using namespace shapelets;
using namespace mpl;

class RProto {
public:
    RProto(std::string _protofile) : protofile(_protofile) {
    }
    Rcpp::List& read() {
        return _read(true);
    }
    Rcpp::List& rawRead() {
        return _read(false);
    }

    Rcpp::List& _read(bool simplify = true) {
        if(values.size() == 0) {
            if(getFileSize(protofile) == 0) return values;

            ifstream f(protofile);
            vector<Ptr<SerializableBase>> obj;


            try {
                Stream str(f, Stream::Binary);

                Ptr<SerializableBase> o = str.readDynamicBase();

                if (!o) {
                    ERR("Can't read protofile " << protofile << "\n");
                }

                while(o) {
                    obj.push_back(o);
                    o = str.readDynamicBase();
                }
            } catch(const std::exception &e) {
                ERR("Can't open " << protofile << " for reading: " <<  e.what() << "\n");
            }

            if((obj.size() == 1)&&(simplify)) {
                values = convertToR(obj[0]);
                obj[0].destroy();
            } else {
                if(obj[0]->name() == "FilterMatch") {
                    values = convertFilterMatches(obj);
                } else {
                    vector<Rcpp::List> ret;
                    for(auto &o: obj) {
                        Rcpp::List l = convertToR(o);
                        if(l.size()>0) {
                            ret.push_back(l);
                        }
                        o.destroy();
                    }
                    values = Rcpp::wrap(ret);
                }
            }
        }
        return values;
    }
    void write(Rcpp::List &o) {
        Ptr<SerializableBase> b = convertFromR<SerializableBase>(o);
        ofstream f(protofile);
        Stream str(f, Stream::Binary);
        str.write(b);
    }

    static Rcpp::List convertFilterMatches(vector<Ptr<SerializableBase>> &obj) {
        vector<double> t;
        vector<size_t> fi;
        vector<double> s;
        for(auto &o: obj) {
            Ptr<FilterMatch> m = o.as<FilterMatch>();
            if(!m) { ERR("Can't cast"); }
            t.push_back(m->t);
            fi.push_back(m->fi);
            s.push_back(m->s);
        }

        return Rcpp::List::create(
            Rcpp::Named("t") = Rcpp::wrap(t),
            Rcpp::Named("fi") = Rcpp::wrap(fi),
            Rcpp::Named("s") = Rcpp::wrap(s)
        );
    }

    static SEXP convertToR(Ptr<SerializableBase> o) {
        Rcpp::List out;
        if(o->name() == "Statistics") {
            Ptr<Statistics> od = o.as<Statistics>();
            if(!od) { ERR("Can't cast"); }

            StatisticsInfo info = od->getInfo();
            for(auto &name: info.stat_names) {
                out[name] = Rcpp::wrap(od->getStats()[name].values);
            }
        }
        if(o->name() == "Subsequence") {
            Ptr<Subsequence> sub = o.as<Subsequence>();
            if(!sub) { ERR("Can't cast"); }

            o = sub->referent().ptr();
        }
        if(o->name() == "TimeSeries") {
            Ptr<TimeSeries> od = o.as<TimeSeries>();
            if(!od) { ERR("Can't cast"); }

            Rcpp::NumericMatrix ts_vals(od->dim(), od->length());
            for(size_t i=0; i<od->data.size(); ++i) {
                for(size_t j=0; j<od->data[i].values.size(); ++j) {
                    ts_vals(i, j) = od->data[i].values[j];
                }
            }
            out = Rcpp::List::create(
                  Rcpp::Named("values") = ts_vals
                , Rcpp::Named("ts_info") = Rcpp::List::create(
                      Rcpp::Named("labels_ids") = Rcpp::wrap(od->info.labels_ids)
                    , Rcpp::Named("unique_labels") = Rcpp::wrap(od->info.unique_labels)
                    , Rcpp::Named("labels_timeline") = Rcpp::wrap(od->info.labels_timeline)
                )
            );
        }
        if(o->name() == "TimeSeriesComplex") {
            Ptr<TimeSeriesComplex> od = o.as<TimeSeriesComplex>();
            if(!od) { ERR("Can't cast"); }

            vector<vector<complex<double>>> ts_vals;
            for(auto &d : od->data) {
                ts_vals.push_back(d.values);
            }
            out = Rcpp::List::create(
                  Rcpp::Named("values") = Rcpp::wrap(ts_vals)
                , Rcpp::Named("ts_info") = Rcpp::List::create(
                      Rcpp::Named("labels_ids") = Rcpp::wrap(od->info.labels_ids)
                    , Rcpp::Named("unique_labels") = Rcpp::wrap(od->info.unique_labels)
                    , Rcpp::Named("labels_timeline") = Rcpp::wrap(od->info.labels_timeline)
                )
            );
        }
        if(o->name() == "SpikesList") {
            Ptr<SpikesList> od = o.as<SpikesList>();
            if(!od) { ERR("Can't cast"); }

            vector<vector<double>> sp;
            for(auto &seq : od->seq) {
                sp.push_back(seq.values);
            }
            out = Rcpp::List::create(
                  Rcpp::Named("values") = Rcpp::wrap(sp)
                , Rcpp::Named("ts_info") = Rcpp::List::create(
                      Rcpp::Named("labels_ids") = Rcpp::wrap(od->ts_info.labels_ids)
                    , Rcpp::Named("unique_labels") = Rcpp::wrap(od->ts_info.unique_labels)
                    , Rcpp::Named("labels_timeline") = Rcpp::wrap(od->ts_info.labels_timeline)
                )
            );
        }
        if(o->name() == "DoubleMatrix") {
            Ptr<DoubleMatrix> m = o.as<DoubleMatrix>();
            if(!m) { ERR("Can't cast"); }

            Rcpp::NumericMatrix rm(m->nrow(), m->ncol());
            for(size_t i=0; i<m->nrow(); ++i) {
                for(size_t j=0; j<m->ncol(); ++j) {
                    rm(i,j) = m->getElement(i,j);
                }
            }
            if(m->uniqueLabels().size()>0) {
                Rcpp::CharacterVector rows(m->rowLabelsIds().size());
                Rcpp::CharacterVector cols(m->colLabelsIds().size());

                for(size_t el_i=0; el_i<m->rowLabelsIds().size(); ++el_i) {
                    size_t lid = m->rowLabelsIds()[el_i];
                    rows(el_i) = m->uniqueLabels()[lid];
                }
                for(size_t el_i=0; el_i<m->colLabelsIds().size(); ++el_i) {
                    size_t lid = m->colLabelsIds()[el_i];
                    cols(el_i) = m->uniqueLabels()[lid];
                }

                rm.attr("dimnames") = Rcpp::List::create(rows, cols);
            }
            return rm;
        }
        Ptr<SpikeNeuronBase> nb = o.as<SpikeNeuronBase>();
        if(nb) {
            vector<double> weights;
            vector<double> ids_pre;
            for(auto &s: nb->getSynapses()) {
                weights.push_back(s.ref().weight());
                ids_pre.push_back(s.ref().idPre());
            }

            out = Rcpp::List::create(
                  Rcpp::Named("xi") = nb->xi()
                , Rcpp::Named("yi") = nb->yi()
                , Rcpp::Named("axon_delay") = nb->axonDelay()
                , Rcpp::Named("id") = nb->id()
                , Rcpp::Named("colSize") = nb->colSize()
                , Rcpp::Named("localId") = nb->localId()
                , Rcpp::Named("synapses") = Rcpp::List::create(
                      Rcpp::Named("weights") = Rcpp::wrap(weights)
                    , Rcpp::Named("ids_pre") = Rcpp::wrap(ids_pre)
                )
            );
        }
        if(o->name() == "MatchingPursuitConfig") {
            Ptr<MatchingPursuitConfig> m = o.as<MatchingPursuitConfig>();
            if(!m) { ERR("Can't cast"); }

            out = Rcpp::List::create(
                Rcpp::Named("threshold") = m->threshold,
                Rcpp::Named("learn_iterations") = m->learn_iterations,
                Rcpp::Named("jobs") = m->jobs,
                Rcpp::Named("learning_rate") = m->learning_rate,
                Rcpp::Named("filters_num") = m->filters_num,
                Rcpp::Named("filter_size") = m->filter_size,
                Rcpp::Named("learn") = m->learn,
                Rcpp::Named("continue_learning") = m->continue_learning,
                Rcpp::Named("batch_size") = m->batch_size,
                Rcpp::Named("seed") = m->seed,
                Rcpp::Named("noise_sd") = m->noise_sd
            );

        }
        if(o->name() == "FilterMatch") {
            Ptr<FilterMatch> m = o.as<FilterMatch>();
            if(!m) { ERR("Can't cast"); }
            out = Rcpp::List::create(
                Rcpp::Named("t") = m->t,
                Rcpp::Named("fi") = m->fi,
                Rcpp::Named("s") = m->s
            );
        }
        out.attr("class") = o->name();

        return out;
    }

    template <typename T, typename CP = FactoryCreationPolicy>
    static Ptr<T> convertFromR(const Rcpp::List &list) {
        string name = list.attr("class");
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
            Ptr<TimeSeries> ts = CP::template create<TimeSeries>();
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
            return ts.as<T>();
        }
        if(name == "SpikesList") {
            Rcpp::List spikes = list["values"];
            Ptr<SpikesList> sl = CP::template create<SpikesList>();

            for(auto &sp_v: spikes) {
                SpikesSequence sp_seq;
                sp_seq.values = Rcpp::as<vector<double>>(sp_v);
                sl->seq.push_back(sp_seq);
            }
            sl->ts_info = ts_info;
            return sl.as<T>();
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
            return r.as<T>();
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

            return c.as<T>();
        }
        if(name == "FilterMatch") {
            Ptr<FilterMatch> m = Factory::inst().createObject<FilterMatch>();
            m->fi = list["fi"];
            m->t = list["t"];
            m->s = list["s"];

            return m.as<T>();
        }
        ERR("Can't convert " << name );
    }



    static vector<FilterMatch> convertFromRFilterMatches(const Rcpp::List &matches_l) {
        vector<FilterMatch> matches;
        Rcpp::NumericVector t = matches_l["t"];
        Rcpp::NumericVector s = matches_l["s"];
        Rcpp::IntegerVector fi = matches_l["fi"];
        for(size_t i=0; i<t.size(); ++i) {
            matches.push_back(
                FilterMatch(fi[i], s[i], t[i])
            );
        }
        return matches;

    }

    void print() {
        cout << "RProto instance. run instance$read() method to read protobuf\n";
    }

    Rcpp::List values;
    std::string protofile;
};



#endif
