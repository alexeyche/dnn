#ifndef RKERNEL_H
#define RKERNEL_H


#include <spikework/kernel.h>
#include <spikework/spikework.h>

#undef PI
#define STRICT_R_HEADERS
#include <Rcpp.h>


#include "RProto.h"
#include "common.h"

using namespace dnn;

class RKernel {
public:
	RKernel() {}

	SEXP run(Rcpp::List data, Rcpp::List l) {
		Spikework::Stack stack;

        if(!data.containsElementNamed("values")) {
        	ERR("Need values in list. It is supposed to be time series or spikes list")
        }
        if(!data.containsElementNamed("ts_info")) {
        	ERR("Need time series information in list named by ts_info. It is supposed to be time series or spikes list")
        }

		KernelWorker w;
		if(l.containsElementNamed("jobs")) {
			w.setJobs(l["jobs"]);	
		}

        SEXP values = data["values"];
        if(Rf_isMatrix(values)) {
        	stack.push( RProto::convertBack(data, "TimeSeries") );
        } else {
        	try {
        		Ptr<SpikesList> sp_l = RProto::convertBack<SpikesList>(data, "SpikesList");
            	stack.push(sp_l->convertToBinaryTimeSeries(1.0));
            	sp_l.destroy();	
        	} catch (...) {
        		ERR("Can't deduce type of input data\n");	
        	}
        }

		
		vector<string> args;
		if(l.containsElementNamed("preprocessor")) {
			SEXP s = l["preprocessor"];
			if(TYPEOF(s) != STRSXP) {
				ERR("Awaiting string in preprocessor option, got " << TYPEOF(s));
			}
			Rcpp::CharacterVector tmp = Rcpp::as<Rcpp::CharacterVector>(s);
            if(tmp.size() != 1) {
                ERR("Need one value in options to set preprocessor\n");
            }
            args.push_back("--preprocessor");
			args.push_back(string(tmp[0]));
		}
		if(l.containsElementNamed("kernel")) {
			SEXP s = l["kernel"];
			if(TYPEOF(s) != STRSXP) {
				ERR("Awaiting string in kernel option, got " << TYPEOF(s));
			}
			Rcpp::CharacterVector tmp = Rcpp::as<Rcpp::CharacterVector>(s);
            if(tmp.size() != 1) {
                ERR("Need one value in options to set kernel\n");
            }
            args.push_back("--kernel");
			args.push_back(string(tmp[0]));
		}
		if(args.size() == 0) {
			ERR("Bad option specification: expect kernel or preprocessor names in a list\n");
		}
		w.processArgs(args);
		w.start(stack);
		w.process(stack);
		w.end(stack);
		
		Ptr<SerializableBase> out = stack.pop();
		return RProto::convertToR(out);
	}

    void print() {
    	cout << "Kernel, preprocessor and kernel functions\n";
    	KernelWorker::descr();
    }
};




#endif