#pragma once

#include <dnn/core.h>
#include <dnn/util/matrix.h>
#include <dnn/util/json.h>
#include <dnn/util/time_series.h>
#include <mpl/protos/mpl.pb.h>
#include <dnn/io/serialize.h>
#include <dnn/util/spikes_list.h>


namespace dnn {
namespace mpl {

class MplInit {
public:
	MplInit();
};



/*@GENERATE_PROTO@*/
struct MatchingPursuitConfig : public Serializable<Protos::MatchingPursuitConfig> {
	MatchingPursuitConfig() :
	  threshold(0.1)
	, learn_iterations(100)
	, jobs(4)
	, learning_rate(1.0)
	, filters_num(100)
	, filter_size(100)
	, learn(true)
	, continue_learning(false)
	, batch_size(1000)
	, seed(-1)
	, noise_sd(0.0)
	{}

	int seed;
	double threshold;
	size_t learn_iterations;
	size_t jobs;
	double learning_rate;
	size_t filters_num;
	size_t filter_size;
	bool learn;
	bool continue_learning;
	size_t batch_size;
	double noise_sd;

	void serial_process() {
        begin() <<
					"threshold: " << threshold << ", " <<
					"learn_iterations: " << learn_iterations << ", " <<
					"jobs: " << jobs << ", " <<
					"learning_rate: " << learning_rate << ", " <<
					"filters_num: " << filters_num << ", " <<
					"filter_size: " << filter_size << ", " <<
					"learn: " << learn << ", " <<
					"continue_learning: " << continue_learning <<
					"batch_size: " << batch_size <<
					"seed: " << seed <<
					"noise_sd: " << noise_sd
				<< Self::end;
	}
};

/*@GENERATE_PROTO@*/
struct FilterMatch : public Serializable<Protos::FilterMatch>  {
	FilterMatch() : fi(0), s(0.0), t(0) {}
	FilterMatch(size_t _fi, double _s, double _t) : fi(_fi), s(_s), t(_t) {}
	void serial_process() {
		begin() << "fi: " << fi << ", " << "s: " << s << ", " << "t: " << t << Self::end;
	}

	size_t fi;
	double s;
	size_t t;
};



class MatchingPursuit {
public:
	MatchingPursuit(const MatchingPursuitConfig &_c);

	struct SubSeqRet {
		vector<FilterMatch> matches;
		DoubleMatrix dfilter;

		vector<double> s;
		vector<double> residual;
		vector<size_t> winners_id;
	};

	vector<double> restore(const vector<FilterMatch> &matches);

	void _restore(const vector<FilterMatch> &matches, vector<double> &restored, size_t from=0) const;

	static SubSeqRet runOnSubSeq(const MatchingPursuit &self, const TimeSeries &ts, size_t dim, size_t from, size_t to);

	struct MPLReturn {
		vector<FilterMatch> matches;
		vector<double> residual;
	};

    MPLReturn run(const TimeSeries &ts, const size_t dim);

	const DoubleMatrix& getFilter() {
		return filter;
	}

	void setFilter(const DoubleMatrix &m) {
		if( (m.nrow() != c.filters_num) || (m.ncol() != c.filter_size)) {
			throw dnnException() << "Got inappropriate to config matrix: need " << c.filters_num << ":" << c.filter_size << " size \n";
		}
		filter = m;

		filter.norm();
	}
	static Ptr<SpikesList> convertMatchesToSpikes(const vector<FilterMatch> &matches);
protected:
	DoubleMatrix filter;
	MatchingPursuitConfig c;
};


}
}