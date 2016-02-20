#ifndef MOD_H
#define MOD_H

#include "RConstants.h"
#include "RSim.h"
#include "RProto.h"
// #include "RMpl.h"
#include "RGammatoneFB.h"
#include "RKernel.h"

#include <Rcpp.h>

class RSim;
class RConstants;
class RProto;
// class RMatchingPursuit;
class RGammatoneFB;
class RKernel;

RCPP_EXPOSED_CLASS(RSim)
RCPP_EXPOSED_CLASS(RConstants)
RCPP_EXPOSED_CLASS(RProto)
// RCPP_EXPOSED_CLASS(RMatchingPursuit)
RCPP_EXPOSED_CLASS(RGammatoneFB)
RCPP_EXPOSED_CLASS(RKernel)

#endif
