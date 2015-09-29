

#include "mpl.h"

#include <dnn/util/log/log.h>
#include <dnn/base/factory.h>

#include <dnn/base/type_deducer.h>

namespace dnn {
namespace mpl {


class MplTypeDeduce : public TypeDeducer {
public:
    string deduceType(const std::type_info &info) const {
        #define REG_FILE <mpl/mpl_register.x>
        #include <dnn/base/deduce_type_impl.x>
        #undef REG_FILE
    }
};


MplInit::MplInit() {
    #define REG_FILE <mpl/mpl_register.x>
    #include <dnn/base/register_impl.x>
    #undef REG_FILE

    Factory::inst().addTypeDeducer(new MplTypeDeduce());
}

MplInit init;


MatchingPursuit::MatchingPursuit(const MatchingPursuitConfig &_c) : c(_c) {
    if(c.seed < 0) {
        std::srand ( unsigned ( std::time(0) ) );
    } else {
        std::srand ( c.seed );
    }
    filter.allocate(c.filters_num, c.filter_size);
    for(size_t i=0; i<c.filters_num; ++i) {
        double acc = 0.0;
        for(size_t j=0; j<c.filter_size; ++j) {
            filter(i, j) = getNorm();
            acc += filter(i, j) * filter(i, j);
        }
        double n = sqrt(acc);
        for(size_t j=0; j<c.filter_size; ++j) {
            filter(i, j) = filter(i, j)/n;
        }
    }
}

vector<double> MatchingPursuit::restore(const vector<FilterMatch> &matches) {
    TimeSeries ts;
    size_t max_t=0;
    for(auto &m: matches) {
        max_t = std::max(max_t, (size_t)m.t);
    }

    vector<double> restored;
    restored.resize(max_t + filter.ncol());
    _restore(matches, restored);
    return restored;
}


void MatchingPursuit::_restore(const vector<FilterMatch> &matches, vector<double> &restored, size_t from) const {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, c.noise_sd);

    for(auto &v: restored) v = 0.0;
    for(auto &m: matches) {
        for(size_t i=0; i<filter.ncol(); ++i) {
            restored[m.t + i - from] += m.s * filter(m.fi, i) + distribution(generator);
        }
    }
    double denom = 0.0;
    for(const auto &v: restored) denom += v*v;
    denom = sqrt(denom);
    for(auto &v: restored) v /= denom;
}

MatchingPursuit::SubSeqRet MatchingPursuit::runOnSubSeq(const MatchingPursuit &self, const TimeSeries &ts, size_t dim, size_t from, size_t to) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, self.c.noise_sd);

    MatchingPursuit::SubSeqRet r;
    if(self.c.learn) {
        r.dfilter.allocate(self.filter.nrow(), self.filter.ncol());
        r.dfilter.fill(0.0);
    }
    for(size_t i=from; i<to; ++i) {
        if(i >= ts.data[dim].values.size()) {
            throw dnnException() << "Trying to get value out of input data: " << dim << ":" << i << "\n";
        }
        r.residual.push_back(ts.data[dim].values[i]);
    }

    for(size_t ti=0; ti<(r.residual.size()-self.filter.ncol()); ++ti) {
        size_t ti_f = ti+self.filter.ncol();

        double x_denom = 0.0;
        for(size_t xi=ti; xi<ti_f; ++xi) {
            x_denom += r.residual[xi]*r.residual[xi];
        }
        x_denom = sqrt(x_denom);
        for(size_t i=0; i<self.c.learn_iterations; ++i) {
            double max_s = -100;
            size_t max_fi = 0;

            for(size_t fi=0; fi<self.filter.nrow(); ++fi) {
                double s_f=0;
                for(size_t xi=ti; xi<ti_f; ++xi) {
                    s_f += (r.residual[xi]/x_denom) * self.filter(fi, xi-ti);
                }

                if (max_s<s_f) {
                    max_s = s_f;
                    max_fi = fi;
                }
            }
            // double noise = distribution(generator);
            double noise = 0.0;
            if(max_s+noise>=self.c.threshold) {
                // L_DEBUG << "noise: " << noise << ", max_s: " << max_s;
                FilterMatch m(max_fi, max_s, ti+from);

                for(size_t xi=ti; xi<ti_f; ++xi) {
                    r.residual[xi] -= x_denom * m.s * self.filter(m.fi, xi-ti);
                }
                r.matches.push_back(m);
            } else {
                break;
            }
        }
    }
    if( (self.c.learn) && (r.matches.size()>0) ) {
        vector<double> restored;
        restored.resize(r.residual.size());
        self._restore(r.matches, restored, from);

        for(auto &m: r.matches) {
            double x_denom = 0.0;
            double y_denom = 0.0;
            for(size_t i=0; i<self.filter.ncol(); ++i) {
                x_denom += ts.data[dim].values[m.t + i] * ts.data[dim].values[m.t + i];
                y_denom += restored[m.t + i - from] * restored[m.t + i - from];
            }
            x_denom = sqrt(x_denom);
            y_denom = sqrt(y_denom);
            for(size_t i=0; i<self.filter.ncol(); ++i) {
                double delta = ts.data[dim].values[m.t + i]/x_denom - restored[m.t + i - from]/y_denom;
                // double delta = ts.data[dim].values[m.t + i] - restored[m.t + i - from];
                r.dfilter(m.fi, i) += m.s * delta;
            }
        }
    }
    return r;
}

MatchingPursuit::MPLReturn MatchingPursuit::run(const TimeSeries &ts, const size_t dim) {
    MatchingPursuit::MPLReturn runret;
    for(size_t bi=0; bi<ts.data[dim].values.size(); bi+=c.batch_size) {
        vector<FilterMatch> matches;
        vector<IndexSlice> slices = dispatchOnThreads(
            std::min(ts.data[dim].values.size()-bi, c.batch_size), c.jobs
        );
        vector<std::future<SubSeqRet>> futures;
        for(auto &slice: slices) {
            futures.push_back(
                std::async(
                    std::launch::async,
                    runOnSubSeq,
                    std::cref(*this),
                    std::cref(ts),
                    dim,
                    bi+slice.from,
                    bi+slice.to
                )
            );
            L_DEBUG << "Running worker on slice " << bi+slice.from << ": " << bi+slice.to;
        }

        vector<SubSeqRet> rets;
        for(auto &fret: futures) {
            SubSeqRet ret = fret.get();
            L_DEBUG << "done";
            for(auto &m: ret.matches) {
                runret.matches.push_back(m);
            }
            for(const auto& r: ret.residual) {
                runret.residual.push_back(r);
            }
            rets.push_back(ret);
        }

        if(c.learn) {
            for(auto &ret : rets) {
                for(size_t i=0; i<filter.nrow(); ++i) {
                    double acc = 0.0;
                    for(size_t j=0; j<filter.ncol(); ++j) {
                        filter(i, j) += c.learning_rate * ret.dfilter(i, j);
                        acc += filter(i, j) * filter(i, j);
                    }
                    double n = sqrt(acc);
                    for(size_t j=0; j<filter.ncol(); ++j) {
                        filter(i, j) = filter(i, j)/n;
                    }
                }
            }
        }
    }
    return runret;
}

Ptr<SpikesList> MatchingPursuit::convertMatchesToSpikes(const vector<FilterMatch> &matches) {
    Ptr<SpikesList> sl = Factory::inst().createObject<SpikesList>();
    for(const auto& m: matches) {
        sl->addSpike(m.fi, m.t);
    }
    return sl;
}


}
}
