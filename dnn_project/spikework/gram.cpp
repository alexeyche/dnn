
#include "gram.h"

#include <dnn/util/time_series.cpp>
#include <dnn/util/matrix.cpp>

namespace dnn {

void GramProcessor::usage() {
    cout << "GramProcessor building gram matrix on input time series\n";
    cout << "   --text,      print to stdout matrix in text representation\n";
    cout << "   --mode,      modes specifying how to deal with multiple dimensions: \n";
    cout << "       mul  (default) multiply inner products of each dimension \n";
    cout << "       acc            sum inner products of each dimension\n";
    cout << "\n";
    IOProcessor::usage();
}

void GramProcessor::processArgs(const vector<string> &args) {
    IOProcessor::processArgs(args);
    OptionParser op(args);
    string mode_str = "mul";
    op.loption("--text", text, false, true);
    op.loption("--mode", mode_str, false);
    if(mode_str == "mul") {
        mode = MUL;
    } else
    if(mode_str == "acc") {
        mode = ACC;
    } else {
        throw dnnException() << "Can't recognize inner product mode: " << mode_str << "\n";
    }
}

void GramProcessor::process(Spikework::Stack &s) {
    Ptr<TimeSeries> ts = s.pop().as<TimeSeries>();

    vector<Ptr<TimeSeries>> ts_chopped = ts->chop();

    Ptr<DoubleMatrix> gram_matrix(Factory::inst().createObject<DoubleMatrix>());
    gram_matrix->allocate(ts_chopped.size(), ts_chopped.size());
    for(size_t i=0; i<ts_chopped.size(); ++i) {
        gram_matrix->setRowLabel(i, ts_chopped[i]->getLabel());
        for(size_t j=0; j<ts_chopped.size(); ++j) {
            gram_matrix->setColLabel(j, ts_chopped[j]->getLabel());
            if(mode == MUL) {
                gram_matrix.ref()(i, j) = ts_chopped[i]->innerProductMul(ts_chopped[j].ref());
            } else
            if(mode == ACC) {
                gram_matrix.ref()(i, j) = ts_chopped[i]->innerProductAcc(ts_chopped[j].ref());
            } else {
                throw dnnException() << "UB";
            }
        }
    }
    if(text) {
        gram_matrix->textRepr(cout);
    }
    s.push(gram_matrix);
}







}