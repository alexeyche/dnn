
#include "dist.h"

#include <dnn/util/matrix.h>
#include <dnn/util/log/log.h>

#include "subsequence.h"
#include "stats.h"

namespace dnn {
namespace shapelets {


double DistAlgorithm::mean(const Subsequence& sub, const Stats &stats, const size_t &dim) {
    const vector<double>& sums = stats.cumulativeSum(sub.id(), dim);
    return (sums[ sub.from()+sub.length() ] - sums[sub.from()] )/sub.length();
}

double DistAlgorithm::meanSquared(const Subsequence& sub, const Stats &stats, const size_t &dim) {
    const vector<double>& sums = stats.squaredCumulativeSum(sub.id(), dim);
    return (sums[ sub.from()+sub.length() ] - sums[sub.from()])/sub.length();
}

double DistAlgorithm::meanProd(const Subsequence& left_sub, const Subsequence& right_sub, const Stats &stats, const size_t &dim) {
    const Ptr<DoubleMatrix> &prod_stat = stats.meanProdMatrix(right_sub.id(), dim);
    return prod_stat->getElement(
        left_sub.from() + left_sub.length(), right_sub.from() + right_sub.length()
    ) - prod_stat->getElement(
        left_sub.from(), right_sub.from()
    );
}

double DistAlgorithm::sdist(const Subsequence &left, const Subsequence &right, const Stats &stats) {
    assert(left.dim() == right.dim());

    double total_dist = 0.0;
    for(size_t di=0; di<left.dim(); ++di) {
        double m_left = mean(left, stats, di);
        double m_right = mean(right, stats, di);
        L_DEBUG << "mean left: " << m_left << " mean right: " << m_right;

        double sq_m_left = meanSquared(left, stats, di);
        double sq_m_right = meanSquared(right, stats, di);
        L_DEBUG << "sq mean left: " << sq_m_left << " sq mean right: " << sq_m_right;

        double mean_prod = meanProd(left, right, stats, di);
        L_DEBUG << "mean_prod: " << mean_prod;
        double cov = mean_prod - m_left * m_right;
        double sd2_left = sq_m_left - m_left * m_left;
        double sd2_right = sq_m_right - m_right * m_right;
        // L_DEBUG << "cov: " << cov << " sd2_left: " << sd2_left << " sd2_right: " << sd2_right;
        double corr;
        if( (fabs(sd2_left) < 1e-09) && (fabs(sd2_right) < 1e-09) ) {
            corr = 1.0;
        }
        else
        if( (fabs(sd2_left) < 1e-09) || (fabs(sd2_right) < 1e-09) ) {
            corr = 0.0;
        }
        else {
            corr = cov/sqrt(sd2_left*sd2_right);
        }
        double dist2 = 2.0 * (1.0 - corr);
        L_INFO << "corr: " << corr << " dist2: " << dist2;

        total_dist += dist2 > 1e-09 ? sqrt(dist2) : 0.0;
    }
    // L_DEBUG << "total_dist: " << total_dist;
    return total_dist;
}

}
}