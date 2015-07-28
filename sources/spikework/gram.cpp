
#include "gram.h"

#include <dnn/util/time_series.cpp>
#include <dnn/util/matrix.cpp>

namespace dnn {

void GramProcessor::usage() {
    cout << "GramProcessor building gram matrix on input time series\n";    
    cout << "   --csv,      print to stdout matrix in csv representation\n";
    cout << "\n";
    IOProcessor::usage();
}

void GramProcessor::processArgs(const vector<string> &args) {
    IOProcessor::processArgs(args);
    OptionParser op(args);
    op.loption("--csv", csv, false, true);
}

void GramProcessor::process(Spikework::Stack &s) {
    Ptr<TimeSeries> ts = s.pop().as<TimeSeries>();
    size_t elem_id = 0;
    vector<Ptr<TimeSeries>> ts_chopped;
    assert(ts->info.labels_timeline.size() == ts->info.labels_ids.size());
    for(size_t li=0; li<ts->info.labels_timeline.size(); ++li) {
        const size_t &end_of_label = ts->info.labels_timeline[li]; 
        const size_t &label_id = ts->info.labels_ids[li];
        const string &label = ts->info.unique_labels[label_id];

        Ptr<TimeSeries> labeled_ts(Factory::inst().createObject<TimeSeries>());
        for(; elem_id < end_of_label; ++elem_id) {       
            for(size_t di=0; di<ts->data.size(); ++di) {
                labeled_ts->addValue(di, ts->data[di].values[elem_id]);
            }
        }
        labeled_ts->info.addLabelAtPos(label, labeled_ts->length());
        ts_chopped.push_back(labeled_ts);
    }
    Ptr<DoubleMatrix> gram_matrix(Factory::inst().createObject<DoubleMatrix>());
    gram_matrix->allocate(ts_chopped.size(), ts_chopped.size());
    for(size_t i=0; i<ts_chopped.size(); ++i) {
        gram_matrix->setRowLabel(i, ts_chopped[i]->getLabel());
        for(size_t j=0; j<ts_chopped.size(); ++j) {
            gram_matrix->setColLabel(j, ts_chopped[j]->getLabel());
            gram_matrix.ref()(i, j) = ts_chopped[i]->innerProduct(ts_chopped[j].ref());
        }
    }
    if(csv) {
        gram_matrix->csvRepr(cout);        
    }
    s.push(gram_matrix);
}

    





}