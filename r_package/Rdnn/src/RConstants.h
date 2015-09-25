#ifndef RCONSTANTS_H
#define RCONSTANTS_H

#include <dnn/base/constants.h>
#include <dnn/util/json.h>


#undef PI
#define STRICT_R_HEADERS
#include <Rcpp.h>

#include <R.h>
#include <Rinternals.h>

#include "common.h"

using namespace rapidjson;
using namespace dnn;

class RConstants {
public:
    RConstants(Constants &_c) : c(_c) {
    }

    void print() {
        cout << c;
    }

    string substituteElem(string json_str, const Rcpp::List &l) {
        Document json = Json::parseString(json_str);

        SEXP names = Rf_getAttrib(l, R_NamesSymbol);
        if(Rf_isNull(names)) {
            Rcpp::stop("Expecting names list as second argument");
        }
        for (size_t i = 0; i < l.size(); ++i) {
            string key = CHAR(STRING_ELT(names, i));

            auto m = json.FindMember(key.c_str());

            if (m == json.MemberEnd()) {
                ERR("Errors while finding field: " << key << "\n");
            }
            Value &json_val = m->value;

            SEXP v = l[key];
            switch( TYPEOF(v) ) {
                case VECSXP: {
                    ERR("Found list as value in " << key << ". Need flat list\n");
                }
                case REALSXP: {
                    Rcpp::NumericVector tmp = Rcpp::as<Rcpp::NumericVector>(v);
                    if(tmp.size() != 1) {
                        ERR("Need one number to set value in " << key << "\n");
                    }

                    if(json_val.IsDouble()) {
                        json_val.SetDouble(tmp[0]);
                    } else
                    if(json_val.IsInt()) {
                        json_val.SetInt(tmp[0]);
                    } else
                    if(json_val.IsUint()) {
                        json_val.SetUint(tmp[0]);
                    } else {
                        ERR("Found value of different type in " << key << "\n");
                    }
                    break;
                }
                case INTSXP: {
                    if(Rf_isFactor(v) ) {
                        ERR("Found factor in list value by name: " << key << "\n");
                    }
                    Rcpp::IntegerVector tmp = Rcpp::as<Rcpp::IntegerVector>(v);
                    if(tmp.size() != 1) {
                        ERR("Need one number to set value in " << key << "\n");
                    }

                    if(json_val.IsDouble()) {
                        json_val.SetDouble(tmp[0]);
                    } else
                    if(json_val.IsInt()) {
                        json_val.SetInt(tmp[0]);
                    } else
                    if(json_val.IsUint()) {
                        json_val.SetUint(tmp[0]);
                    } else {
                        ERR("Found value of different type in " << key << "\n");
                    }

                    break;
                }
                case STRSXP: {
                    if(!json_val.IsString()) {
                        ERR("Found value of different type in " << key << "\n");
                    }
                    Rcpp::CharacterVector tmp = Rcpp::as<Rcpp::CharacterVector>(v);
                    if(tmp.size() != 1) {
                        ERR("Need one number to set value in " << key << "\n");
                    }
                    string vstr(tmp[0]);
                    json_val.SetString(vstr.c_str(), Json::d.GetAllocator());
                    break;
                }
                case LGLSXP: {
                    if(!json_val.IsBool()) {
                        ERR("Found value of different type in " << key << "\n");
                    }
                    Rcpp::LogicalVector tmp = Rcpp::as<Rcpp::LogicalVector>(v);
                    if(tmp.size() != 1) {
                        ERR("Need one number to set value in " << key << "\n");
                    }
                    json_val.SetBool(tmp[0]);
                    break;
                }
                default: {
                    Rcpp::stop("incompatible SEXP encountered; only accepts lists containing lists, REALSXPs, and INTSXPs");
                }
            }
        }
        return Json::stringify(json);
    }

    string getRidOfExtraKeys(string json_spec, const Rcpp::List &list_spec) {
        Document json_spec_json = Json::parseString(json_spec);
        Value::MemberIterator itr = json_spec_json.MemberBegin();
        while(itr != json_spec_json.MemberEnd()) {
            if (!list_spec.containsElementNamed(itr->name.GetString())) {
                itr = json_spec_json.RemoveMember(itr);
            } else {
                ++itr;
            }
        }
        return Json::stringify(json_spec_json);
    }
    void checkIfExists(string spec, string key, const map<string, string> &consts) {
        Document spec_json = Json::parseString(spec);
        auto ptr = spec_json.FindMember(key.c_str());
        if(ptr != spec_json.MemberEnd()) {
            Value &json_val = ptr->value;
            string name = json_val.GetString();
            auto c_ptr = consts.find(name);
            if(c_ptr == consts.end()) {
                ERR("Can't find in constants specification of element with name " << name <<"\n");
            }
        }

    }
    void addLayer(Rcpp::List layer_spec) {
        if(!layer_spec.containsElementNamed("size")) {
            ERR("Need size in layer specification\n");
        }
        if(!layer_spec.containsElementNamed("neuron")) {
            ERR("Need neuron in layer specification\n");
        }
        // if(!layer_spec.containsElementNamed("act_function")) {
        //     ERR("Need act_function in layer specification\n");
        // }

        string layer_conf = SimConfiguration::getDeaultLayerConf();
        layer_conf = substituteElem(layer_conf, layer_spec);
        layer_conf = getRidOfExtraKeys(layer_conf, layer_spec);

        checkIfExists(layer_conf, "neuron", c.neurons);
        checkIfExists(layer_conf, "learning_rule", c.learning_rules);
        checkIfExists(layer_conf, "act_function", c.act_functions);
        checkIfExists(layer_conf, "weight_normalization", c.weight_normalizations);

        c.sim_conf.layers.push_back(layer_conf);
    }

    void addConnection(size_t from, size_t to, Rcpp::List conn_spec) {
        if(!conn_spec.containsElementNamed("type")) {
            ERR("Need type in connection specification\n");
        }
        if(!conn_spec.containsElementNamed("synapse")) {
            ERR("Need synapse in connection specification\n");
        }
        if(!conn_spec.containsElementNamed("start_weight")) {
            ERR("Need start_weight in connection specification\n");
        }
        if(c.sim_conf.layers.size() < from) {
            ERR("Failed to connect layers from " << from << ", because there is no such amount of layers\n");
        }
        if(c.sim_conf.layers.size() < to) {
            ERR("Failed to connect layers to " << to << ", because there is no such amount of layers\n");
        }

        string conn_conf = SimConfiguration::getDeaultConnectionConf();
        conn_conf = substituteElem(conn_conf, conn_spec);
        conn_conf = getRidOfExtraKeys(conn_conf, conn_spec);

        checkIfExists(conn_conf, "type", c.connections);
        checkIfExists(conn_conf, "synapse", c.synapses);
        checkIfExists(conn_conf, "inh_synapse", c.synapses);

        c.sim_conf.conn_map.insert( std::make_pair(std::make_pair(from, to), conn_conf) );
    }

    void setElement(string name, Rcpp::List l) {
        bool found = false;
        #define FIND(section) \
            {\
                auto sec_ptr = section.find(name); \
                if(sec_ptr != section.end()) { \
                    section[name] = substituteElem(sec_ptr->second, l);\
                    found = true; \
                }\
            }\

        FIND(c.neurons);
        FIND(c.act_functions);
        FIND(c.synapses);
        FIND(c.inputs);
        FIND(c.learning_rules);
        FIND(c.weight_normalizations);
        FIND(c.connections);

        if(!found) {
            ERR("Can't find specification of " << name << " in const\n");
        }
    }

private:
    Constants &c;

};

#endif

