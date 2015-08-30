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
		*RProto::convertBack<MatchingPursuitConfig>(conf, "MatchingPursuitConfig")
	  ) {

	}
	Rcpp::List run(const Rcpp::NumericVector ts_m) {
		Factory::inst().registrationOff();
		TimeSeries *ts = RProto::convertBack<TimeSeries>(
			Rcpp::List::create(
				Rcpp::Named("values") = ts_m
			),
			"TimeSeries"
		);
		MatchingPursuit::MPLReturn ret = MatchingPursuit::run(*ts, 0);
		delete ts;
		Factory::inst().registrationOff();

		vector<SerializableBase*> vv;
		for(auto &m: ret.matches) {
			vv.push_back(&m);
		}
		Rcpp::List matches_l = RProto::convertFilterMatches(vv);

		Factory::inst().registrationOff();
		Ptr<SpikesList> sl = MatchingPursuit::convertMatchesToSpikes(ret.matches);
		Rcpp::List spikes_l = RProto::convertToList(sl.ptr());
		delete sl.ptr();
		Factory::inst().registrationOn();

		return Rcpp::List::create(
			Rcpp::Named("matches") = matches_l
		  ,	Rcpp::Named("spikes") = spikes_l
		  , Rcpp::Named("residual") = ret.residual
		);
	}
	void setFilter(const Rcpp::NumericMatrix m) {
		MatchingPursuit::setFilter(
			*RProto::convertBack<DoubleMatrix>(
				Rcpp::List::create(
					Rcpp::Named("DoubleMatrix") = m
				),
				"DoubleMatrix"
			)
		);
	}

	void setConf(const Rcpp::List conf) {
		Factory::inst().registrationOff();
		MatchingPursuitConfig *in_c = RProto::convertBack<MatchingPursuitConfig>(conf, "MatchingPursuitConfig");
		MatchingPursuit::c = *in_c;
		delete in_c;
		Factory::inst().registrationOn();
	}

	Rcpp::NumericMatrix getFilter() {
		return RProto::convertToList(&filter)[0];
	}

	Rcpp::NumericVector restore(const Rcpp::List matches_l) {
		vector<FilterMatch> matches = RProto::convertBackFilterMatches(matches_l);
		return Rcpp::wrap(MatchingPursuit::restore(matches));
	}
	void print() {
        cout << "MatchingPursuit instance.\n";
        cout << "config:\n";
        Stream(cout, Stream::Text).writeObject(&c);
    }
};


#endif