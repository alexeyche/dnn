#include "proto.h"

#include <ground/ts/time_series.h>
#include <ground/ts/spikes_list.h>
#include <ground/matrix.h>
#include <ground/stat_gatherer.h>
#include <ground/serial/bin_serial.h>
#include <dnn/spikework/protos/spikework_config.pb.h>

#include <dnn/protos/config.pb.h>

template <>
SEXP TProto::Translate<TTimeSeriesInfo>(const TTimeSeriesInfo& ent) {
    Rcpp::List ret;
    for(const auto& lab_start_info: ent.Labels) {
        const auto& lab_name = ent.UniqueLabelNames[lab_start_info.LabelId];
        ret.push_back(
            Rcpp::List::create(
                Rcpp::Named("label") = lab_name
              , Rcpp::Named("start_time") = lab_start_info.From
              , Rcpp::Named("duration") = lab_start_info.To - lab_start_info.From
            )
        );
    }
    ret.attr("class") = "TimeSeriesInfo";
    return ret;
}


template <>
SEXP TProto::Translate<TDoubleMatrix>(const TDoubleMatrix& ent) {
    Rcpp::NumericMatrix rm(ent.RowSize(), ent.ColSize());
    for(size_t i=0; i<ent.RowSize(); ++i) {
        for(size_t j=0; j<ent.ColSize(); ++j) {
            rm(i,j) = ent(i,j);
        }
    }
    if(ent.GetUniqueLabels().size()>0) {
        Rcpp::CharacterVector rows(ent.GetRowLabelsIds().size());
        Rcpp::CharacterVector cols(ent.GetColLabelsIds().size());

        for(size_t el_i=0; el_i<ent.GetRowLabelsIds().size(); ++el_i) {
            size_t lid = ent.GetRowLabelsIds()[el_i];
            rows(el_i) = ent.GetUniqueLabels()[lid];
        }
        for(size_t el_i=0; el_i<ent.GetColLabelsIds().size(); ++el_i) {
            size_t lid = ent.GetColLabelsIds()[el_i];
            cols(el_i) = ent.GetUniqueLabels()[lid];
        }

        rm.attr("dimnames") = Rcpp::List::create(rows, cols);
    }
    return rm;
}


template <>
SEXP TProto::Translate<TTimeSeries>(const TTimeSeries& ent) {
    Rcpp::NumericMatrix ts_vals(ent.Dim(), ent.Length());
    for(size_t i=0; i<ent.Data.size(); ++i) {
        for(size_t j=0; j<ent.Data[i].Values.size(); ++j) {
            ts_vals(i, j) = ent.Data[i].Values[j];
        }
    }
    Rcpp::List ret = Rcpp::List::create(
          Rcpp::Named("values") = ts_vals
        , Rcpp::Named("info") = Translate<TTimeSeriesInfo>(ent.Info)
    );
    ret.attr("class") = "TimeSeries";
    return ret;
}

template <>
SEXP TProto::Translate<TSpikesList>(const TSpikesList& ent) {
    TVector<TVector<double>> sp;
    for(auto &seq : ent.Data) {
        sp.push_back(seq.Values);
    }

    Rcpp::List ret = Rcpp::List::create(
          Rcpp::Named("values") = Rcpp::wrap(sp)
        , Rcpp::Named("info") = Translate<TTimeSeriesInfo>(ent.Info)
    );
    ret.attr("class") = "SpikesList";
    return ret;
}


template <>
SEXP TProto::Translate<TStatistics>(const TStatistics& ent) {
    Rcpp::List ret = Rcpp::List::create(
          Rcpp::Named("values") = Rcpp::wrap(ent.Values)
        , Rcpp::Named("name") = ent.Name
        , Rcpp::Named("from") = ent.From
        , Rcpp::Named("to") = ent.To
    );
    ret.attr("class") = "Statistics";
    return ret;
}


template <>
TTimeSeriesInfo TProto::TranslateBack<TTimeSeriesInfo>(const Rcpp::List& l) {
    TTimeSeriesInfo ret;
    for(size_t li=0; li<l.size(); ++li) {
        Rcpp::List elem(l[li]);
        ret.AddLabelAtPos(elem["label"], elem["start_time"], elem["duration"]);
    }
    return ret;
}

template <>
TTimeSeries TProto::TranslateBack<TTimeSeries>(const Rcpp::List& l) {
    TTimeSeries ts;
    SEXP values = l["values"];
    if(Rf_isMatrix(values)) {
        Rcpp::NumericMatrix m(values);
        ts.Info.DimSize = m.nrow();
        ts.Data.resize(ts.Info.DimSize);
        for(size_t i=0; i<m.nrow(); ++i) {
            for(size_t j=0; j<m.ncol(); ++j) {
                ts.Data[i].Values.push_back(m(i,j));
            }
        }
    } else {
        ts.Info.DimSize = 1;
        ts.Data.resize(ts.Info.DimSize);
        ts.Data[0].Values = Rcpp::as<std::vector<double>>(values);
    }
    if(l.containsElementNamed("info")) {
        ui32 dimsize = ts.Info.DimSize;
        ts.Info = TranslateBack<TTimeSeriesInfo>(l["info"]);
        ts.Info.DimSize = dimsize;
    }
    return ts;
}

template <>
TSpikesList TProto::TranslateBack<TSpikesList>(const Rcpp::List& l) {
    TSpikesList sl;
    Rcpp::List spikes = l["values"];
    for(auto &sp_v: spikes) {
        TSpikesListData sp_seq;
        sp_seq.Values = Rcpp::as<TVector<double>>(sp_v);
        sl.Data.push_back(sp_seq);
    }

    sl.Info.DimSize = sl.Data.size();

    if(l.containsElementNamed("info")) {
        sl.Info = TranslateBack<TTimeSeriesInfo>(l["info"]);
    }
    return sl;
}

#define SET(config, list, field, type,  name, subfield) \
    { \
        type v; \
        if (GetFromList<type>(list, name, v)) { \
            config.mutable_## field()->set_## subfield(v); \
        }\
    } \

template <>
NDnnProto::TPreprocessorConfig TProto::TranslateBack<NDnnProto::TPreprocessorConfig>(const Rcpp::List& l) {
    NDnnProto::TPreprocessorConfig config;
    if (l.containsElementNamed("Epsp")) {
        *config.mutable_epsp() = NDnnProto::TEpspOptions();

        SET(config, l["Epsp"], epsp, double, "TauRise", taurise);
        SET(config, l["Epsp"], epsp, double, "TauDecay", taudecay);
        SET(config, l["Epsp"], epsp, double, "Length", length);
        SET(config, l["Epsp"], epsp, double, "Dt", dt);
    }
    if (l.containsElementNamed("Gauss")) {
        *config.mutable_gauss() = NDnnProto::TGaussOptions();

        SET(config, l["Gauss"], gauss, double, "Sigma", sigma);
        SET(config, l["Gauss"], gauss, double, "Length", length);
        SET(config, l["Gauss"], gauss, double, "Dt", dt);
    }
    return config;
}



template <>
NDnnProto::TKernelConfig TProto::TranslateBack<NDnnProto::TKernelConfig>(const Rcpp::List& l) {
    NDnnProto::TKernelConfig config;
    if (l.containsElementNamed("Dot")) {
        *config.mutable_dot() = NDnnProto::TDotOptions();
    }
    if (l.containsElementNamed("RbfDot")) {
        *config.mutable_rbfdot() = NDnnProto::TRbfDotOptions();   

        SET(config, l["RbfDot"], rbfdot, double, "Sigma", sigma);
    }
    if (l.containsElementNamed("AnovaDot")) {
        *config.mutable_anovadot() = NDnnProto::TAnovaDotOptions();   

        SET(config, l["AnovaDot"], anovadot, double, "Sigma", sigma);
        SET(config, l["AnovaDot"], anovadot, double, "Power", power);
    }
    if (l.containsElementNamed("Shoe")) {
        *config.mutable_shoe() = NDnnProto::TShoeOptions();
        
        SET(config, l["Shoe"], shoe, double, "Sigma", sigma);
        
        Rcpp::List shoeList = l["Shoe"];
        if (shoeList.containsElementNamed("Kernel")) {
            *(config.mutable_shoe()->mutable_kernel()) = TProto::TranslateBack<NDnnProto::TKernelConfig>(shoeList["Kernel"]);
        }
    }
    return config;
}

#undef SET

Rcpp::List TProto::TranslateModel(const NDnnProto::TConfig& config) {
    Rcpp::List res;
    ui32 neuronId = 0;
    for (const auto& layer: config.layer()) {
        ui32 layerSynapseCounter = 0;
        for (const auto& neuronInner: layer.spikeneuronimplinnerstate()) {
            Rcpp::NumericVector weights;
            Rcpp::NumericVector postSynWeights;
            Rcpp::IntegerVector ids_pre;
            
            for (ui32 synId=0; synId < neuronInner.synapsessize(); ++synId, ++layerSynapseCounter) {
                auto synInner = layer.synapseinnerstate(layerSynapseCounter);
                weights.push_back(synInner.weight());
                postSynWeights.push_back(synInner.postsynapticweight());
                ids_pre.push_back(synInner.idpre());
            }
            res.push_back(Rcpp::List::create(
                Rcpp::Named("synapses") = Rcpp::List::create(
                    Rcpp::Named("weights") = weights,
                    Rcpp::Named("post_synaptic_weights") = postSynWeights,
                    Rcpp::Named("ids_pre") = ids_pre
                ),
                Rcpp::Named("id") = neuronId++
            ));
        }    
    }
    
    return res;
}


Rcpp::List TProto::ReadFromFile(TString protofile) {
    std::ifstream input(protofile, std::ios::binary);
    TBinSerial serial(input);

    Rcpp::List l;
    switch (serial.ReadProtobufType()) {
        case EProto::TIME_SERIES:
            l = Translate(serial.ReadObject<TTimeSeries>());
            break;
        case EProto::SPIKES_LIST:
            l = Translate(serial.ReadObject<TSpikesList>());
            break;
        case EProto::STATISTICS:
            {
                TStatistics stat;
                while (serial.ReadObject<TStatistics>(stat)) {
                    Rcpp::List subList;
                    subList = Translate(stat);
                    l.push_back(subList);
                }
            }
            break;
        case EProto::CONFIG:
            {
                NDnnProto::TConfig config;
                if (!serial.ReadProtobufMessage(config)) {
                    ERR("Failed to read config protobuf: " << protofile);
                }
                l = TranslateModel(config);
            }
            break;
        default:
            ERR("Unknown protobuf type " << protofile);
    }
    return l;
}

void TProto::WriteToFile(Rcpp::List l, TString protofile) {
    std::ofstream output(protofile, std::ios::binary);
    TBinSerial serial(output);

    TString name = l.attr("class");
    if (name == "TimeSeries") {
        serial.WriteObject(TranslateBack<TTimeSeries>(l));
        return;
    }
    if (name == "SpikesList") {
        serial.WriteObject(TranslateBack<TSpikesList>(l));
        return;
    }
    ERR("Failed to find appropriate entity for data in R structure with class " << name);
}

Rcpp::NumericMatrix TProto::ReadModelWeights(TString protofile) {
    std::ifstream input(protofile, std::ios::binary);
    TBinSerial serial(input);

    if (serial.ReadProtobufType() != EProto::CONFIG) {
        ERR("Need model config as input");
    }

    Rcpp::List l;
    NDnnProto::TConfig config;
    if (!serial.ReadProtobufMessage(config)) {
        ERR("Failed to read config protobuf: " << protofile);
    }

    ui32 net_size = 0;
    for (const auto& layer: config.layer()) {
        net_size += layer.spikeneuronimplinnerstate_size();
    }
 
    Rcpp::NumericMatrix weights(net_size, net_size);

    ui32 neuronId = 0;
    for (const auto& layer: config.layer()) {
        ui32 layerSynapseCounter = 0;
        for (const auto& neuronInner: layer.spikeneuronimplinnerstate()) {
            for (ui32 synId=0; synId < neuronInner.synapsessize(); ++synId, ++layerSynapseCounter) {
                auto synInner = layer.synapseinnerstate(layerSynapseCounter);
                weights(neuronId, synInner.idpre()) = synInner.weight();
            }
            ++neuronId;
        }
    }

    return weights;
}