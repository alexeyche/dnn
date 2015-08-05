#pragma once

#include "io_processor.h"


namespace dnn {

class GramProcessor : public IOProcessor {
public:
    enum EInnerProductMode { MUL, ACC };
    GramProcessor() : text(false) {}

    void usage();
    void processArgs(const vector<string> &args);
    void process(Spikework::Stack &s);
private:
    bool text;
    EInnerProductMode mode;
};


}

