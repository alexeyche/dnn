#ifndef RPROTO_READ_H
#define RPROTO_READ_H

#include <ground/base/base.h>
#include <dnn/protos/config.pb.h>

#include <fstream>

#undef PI
#define STRICT_R_HEADERS
#include <Rcpp.h>

#include "common.h"

using namespace NGround;

template <typename T>
bool GetFromList(const Rcpp::List& l, const TString name, T& dst) {
    SEXP names = Rf_getAttrib(l, R_NamesSymbol);
    if (Rf_isNull(names) ) {
        return false;  
    } 
    R_xlen_t n = Rf_xlength(names) ;
    for (R_xlen_t i=0; i<n; i++) {
        if(name == CHAR(STRING_ELT(names, i))) {
            dst = Rcpp::as<T>(l[i]);
            return true;
        }
    }
    return false;
}

class TProto {
public:

    template <typename T>
    static SEXP Translate(const T& ent);

    static Rcpp::List TranslateModel(const NDnnProto::TConfig& config);


    template <typename T>
    static T TranslateBack(const Rcpp::List& l);

    template <typename T>
    void WriteEntity(T&& v, std::ostream& ostr) {
        typename T::TProto pb = v.Serialize();
        pb.SerializeToOstream(&ostr);
    }

    Rcpp::List ReadFromFile(TString protofile);


    Rcpp::NumericMatrix ReadModelWeights(TString protofile);

    void WriteToFile(Rcpp::List l, TString protofile);

    void Print() {
        std::cout << "RProto instance. run instance$read() method to read protobuf\n";
    }

};

#endif