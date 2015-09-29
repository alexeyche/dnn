
#undef PI
#define STRICT_R_HEADERS
#include <Rcpp.h>

#include <dnn/util/log/log.h>
#include <dnn/base/factory.h>

#include "common.h"
#include "util.h"


using namespace dnn;



// [[Rcpp::export]]
void setVerboseLevel(int level) {
    if(level == 0) {
        Log::inst().setLogLevel(Log::INFO_LEVEL);
    } else
    if(level == 1) {
        Log::inst().setLogLevel(Log::DEBUG_LEVEL);
    } else {
        ERR("Available verbose levels : 0 -- INFO, 1 -- DEBUG");
    }
}


// [[Rcpp::export]]
void dnnCleanHeap() {
    Factory::inst().cleanHeap();
}