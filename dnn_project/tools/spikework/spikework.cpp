

#include <spikework/spikework.h>


int main(int argc, char **argv) {
    vector<string> args;
    for(size_t i=1; i<argc; ++i) {
        args.push_back(argv[i]);
    }
    dnn::Spikework sw(args);

    return 0;
}
