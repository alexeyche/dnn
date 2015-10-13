#ifndef RMPL_H
#define RMPL_H


#include <mpl/mpl.h>

#undef PI
#define STRICT_R_HEADERS
#include <Rcpp.h>

#include "RSim.h"

using namespace dnn;
//using namespace mpl;

class RMatchingPursuit : public MatchingPursuit {
public:
	RMatchingPursuit(const Rcpp::List conf)
	: MatchingPursuit(
		*RProto::convertFromR<MatchingPursuitConfig>(conf)
	  ) {

	}
	Rcpp::List run(const Rcpp::NumericVector ts_m) {
		Ptr<TimeSeries> ts = RProto::convertFromR<TimeSeries>(
			Rcpp::List::create(
				Rcpp::Named("values") = ts_m
			)
		);
		MatchingPursuit::MPLReturn ret = MatchingPursuit::run(ts.ref(), 0);

		vector<Ptr<SerializableBase>> vv;
		for(auto &m: ret.matches) {
			vv.push_back(Ptr<SerializableBase>(&m));
		}
		Rcpp::List matches_l = RProto::convertFilterMatches(vv);

		Ptr<SpikesList> sl = MatchingPursuit::convertMatchesToSpikes(ret.matches);
		Rcpp::List spikes_l = RProto::convertToR(sl.ptr());

		return Rcpp::List::create(
			Rcpp::Named("matches") = matches_l
		  ,	Rcpp::Named("spikes") = spikes_l
		  , Rcpp::Named("residual") = ret.residual
		);
	}
	void setFilter(const Rcpp::NumericMatrix m) {
		Ptr<DoubleMatrix> dm = RProto::convertFromR<DoubleMatrix>(
			Rcpp::List::create(
				Rcpp::Named("values") = m
			)
		);
		MatchingPursuit::setFilter(
			*dm
		);
	}

	void setConf(const Rcpp::List conf) {
		Ptr<MatchingPursuitConfig> in_c = RProto::convertFromR<MatchingPursuitConfig>(conf);
		MatchingPursuit::c = in_c.ref();
	}

	Rcpp::NumericMatrix getFilter() {
		return RProto::convertToR(&filter);
	}

	Rcpp::NumericVector restore(const Rcpp::List matches_l) {
		vector<FilterMatch> matches = RProto::convertFromRFilterMatches(matches_l);
		return Rcpp::wrap(MatchingPursuit::restore(matches));
	}
	void print() {
        cout << "MatchingPursuit instance.\n";
        cout << "config:\n";
        Stream(cout, Stream::Text).writeObject(&c);
    }
};


#endif