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
    void write(Rcpp::List &o, const string& name) {
        Ptr<SerializableBase> b = convertBack(o, name);
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
        if(o->name() == "Statistics") {
            Rcpp::List out;
            Ptr<Statistics> od = o.as<Statistics>();
            if(!od) { ERR("Can't cast"); }

            StatisticsInfo info = od->getInfo();
            for(auto &name: info.stat_names) {
                out[name] = Rcpp::wrap(od->getStats()[name].values);
            }
            return out;
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
            return Rcpp::List::create(
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
            return Rcpp::List::create(
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
            return Rcpp::List::create(
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

            return Rcpp::List::create(
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

            return Rcpp::List::create(
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
            return Rcpp::List::create(
                Rcpp::Named("t") = m->t,
                Rcpp::Named("fi") = m->fi,
                Rcpp::Named("s") = m->s
            );
        }
        ERR("Can' recognize " << o->name() << "\n");
    }

    template <typename T>
    static Ptr<T> convertBack(const Rcpp::List &list, const string &name) {
        Ptr<SerializableBase> o = convertBack(list, name);
        Ptr<T> oc = o.as<T>();
        if(!oc) { ERR("Can't cast"); }
        return oc;
    }


    static Ptr<SerializableBase> convertBack(const Rcpp::List &list, const string &name);

    static vector<FilterMatch> convertBackFilterMatches(const Rcpp::List &matches_l) {
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
