
#include "kernel.h"
#include "kernel_factory.h"

#include <dnn/util/util.h>
#include <dnn/util/matrix.h>
#include <dnn/base/factory.h>

namespace dnn {

Ptr<TimeSeries> FunKernel::generate(size_t dim, size_t length, double dt) const  {
    if(fun == nullptr) {
        throw dnnException() << "Need to specify kernel function before generating\n";
    }
    Ptr<TimeSeries> out(Factory::inst().createObject<TimeSeries>());
    for(size_t di=0; di<dim; ++di) {
        double max_t = length * dt;
        for(double s=0; s<max_t; s+=dt) {
            double v = fun(s);
            out->addValue(di, v);
        }
    }
    return out;
}

KernelWorker::~KernelWorker() {
    if(kernel) {
        delete kernel.ptr();
    }
    if(kernel_proc) {
        delete kernel_proc.ptr();
    }
}




void KernelWorker::usage() {
    cout << "KernelWorker applying specifed kernel\n";
    cout << "\t--kernel,  -k        kernel specification (required)\n";
    cout << "\t--preprocessor,  -p  preprocessor specification\n";
    cout << "\t--text FILE,      print to file (- for stdout) kernel matrix in text representation\n";
    cout << "\n";
    cout << "available kernels:\n";
    for(const auto &k: KernelFactory::inst().getKernelsMap()) {
        cout << "\t" <<  k.first << "\n";
        stringstream ss;
        k.second()->usage(ss);
        for(auto l: split(ss.str(), '\n')) {
            cout << "\t\t" << l << "\n";
        }
    }
    cout << "\n";
    cout << "available preprocessors:\n";
    for(const auto &k:  KernelFactory::inst().getKernelPreprocessorsMap()) {
        cout << "\t" <<  k.first << "\n";
        stringstream ss;
        k.second()->usage(ss);
        for(auto l: split(ss.str(), '\n')) {
            cout << "\t\t" << l << "\n";
        }
    }
    cout << "\n";

    IOWorker::usage();
}

void KernelWorker::processArgs(vector<string> &args) {
    IOWorker::processArgs(args);
    OptionParser op(args);
    string kernel_spec;
    string kernel_preproc_spec;

    op.option("--kernel", "-k", kernel_spec, false);
    op.option("--preprocessor", "-p", kernel_preproc_spec, false);
    op.loption("--text", text_file, false);
    op.checkEmpty();

    if(!kernel_spec.empty()) {
        kernel = KernelFactory::inst().createKernel(kernel_spec);
    }
    if(!kernel_preproc_spec.empty()) {
        kernel_proc = KernelFactory::inst().createKernelPreprocessor(kernel_preproc_spec);
    }

    if((!kernel)&&(!kernel_proc)) {
        throw dnnException() << "Can't recognize kernel specification: " << kernel_spec << "\n";
    }
}

void KernelWorker::start(Spikework::Stack &s) {
    IOWorker::start(s);
}

void KernelWorker::process(Spikework::Stack &s) {
    if(kernel_proc) {
        L_DEBUG << "KernelWorker, Start preprocessing";
        IKernelPreprocessor &k = kernel_proc.ref();
        Ptr<TimeSeries> ts = s.pop().as<TimeSeries>();
        Ptr<TimeSeries> ts_out = k(ts);
        s.push(ts_out);
        L_DEBUG << "KernelWorker, End preprocessing";
    }
    if(kernel) {
        L_DEBUG << "KernelWorker, Start applying kernel";
        IKernel &k = kernel.ref();

        Ptr<TimeSeries> ts = s.pop().as<TimeSeries>();

        vector<Ptr<TimeSeries>> ts_chopped = ts->chop();
        if(ts_chopped.size() == 0) {
            throw dnnException() << "Got zero sized time series list, check presence of time series information\n";
        }
        Ptr<DoubleMatrix> gram_matrix(Factory::inst().createObject<DoubleMatrix>());
        DoubleMatrix &m = gram_matrix.ref();
        m.allocate(ts_chopped.size(), ts_chopped.size());
        for(size_t i=0; i<ts_chopped.size(); ++i) {
            m.setRowLabel(i, ts_chopped[i]->getLabel());
            for(size_t j=0; j<i; ++j) {
                m(i, j) = m(j, i);
            }
            for(size_t j=i; j<ts_chopped.size(); ++j) {
                if(i == 0) {
                    m.setColLabel(j, ts_chopped[j]->getLabel());
                }
                m(i, j) = k.process(ts_chopped[i], ts_chopped[j]);
            }
        }
        L_DEBUG << "KernelWorker, End applying kernel";

        if(!text_file.empty()) {
            if(text_file == "-") {
                m.textRepr(cout);
            } else {
                ofstream f(text_file);
                if(!f.good()) {
                    throw dnnException() << "Can't open " << text_file << "\n";
                }
                m.textRepr(f);
            }
        }
        s.push(gram_matrix);
    }
}




}