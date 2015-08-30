
#include "gram.h"

#include <dnn/util/time_series.cpp>
#include <dnn/util/matrix.cpp>

namespace dnn {

void GramWorker::usage() {
    cout << "GramWorker building gram matrix on input time series\n";
    cout << "   --text FILE,      print to file (- for stdout) matrix in text representation\n";
    cout << "   --mode,           modes specifying how to deal with multiple dimensions: \n";
    cout << "       acc (default) sum inner products of each dimension\n";
    cout << "       mul           multiply inner products of each dimension \n";
    cout << "\n";
    IOWorker::usage();
}

void GramWorker::processArgs(vector<string> &args) {
    IOWorker::processArgs(args);
    OptionParser op(args);
    string mode_str;
    op.loption("--text", text_file, false);
    op.loption("--mode", mode_str, false);
    if((mode_str == "acc")|| mode_str.empty()) {
        mode = ACC;
    } else
    if(mode_str == "mul") {
        mode = MUL;
    } else {
        throw dnnException() << "Can't recognize inner product mode: " << mode_str << "\n";
    }
}

void GramWorker::process(Spikework::Stack &s) {
    Ptr<TimeSeries> ts = s.pop().as<TimeSeries>();

    vector<Ptr<TimeSeries>> ts_chopped = ts->chop();
    if(ts_chopped.size() == 0) {
        throw dnnException() << "Got zero sized time series list, check presence of time series information\n";
    }
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
    if(!text_file.empty()) {
        if(text_file == "-") {
            gram_matrix->textRepr(cout);
        } else {
            ofstream f(text_file);
            if(!f.good()) {
                throw dnnException() << "Can't open " << text_file << "\n";
            }
            gram_matrix->textRepr(f);
        }
    }
    s.push(gram_matrix);
}







}