#pragma once

#include "io_processor.h"


namespace dnn {
   
class GramProcessor : public IOProcessor {
public:
    GramProcessor() : csv(false) {}
    
    void usage();
    void processArgs(const vector<string> &args);    
    void process(Spikework::Stack &s);
private:
    bool csv;
};


}

