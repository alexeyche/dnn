
#include <fstream>

#include <dnn/util/option_parser.h>
#include <dnn/util/log/log.h>
#include <dnn/util/time_series.h>

#include <shapelets/shapelets_algo.h>
#include <shapelets/dataset.h>

using namespace dnn;
using namespace shapelets;

const char * usage = R"USAGE(
USAGE: ./shapelets  [options]
Options:
    --input,  -i    Input time series
    --config, -c    Config of shapelets in json
    --verbose, -v   Turn on verbose logging
    --no-colors, -nc   No colors while logging
%s
)USAGE";

void printHelp() {
    ShapeletsConfig c;
    ostringstream s;
    Stream(s, Stream::Text).writeObject(&c);
    string json_str = s.str();
    printf(usage, json_str.c_str());
}

int main(int argc, char **argv) {
    string input_file;
    string config_file;
    bool need_help = false;
    bool verbose = false;
    bool no_colors = false;

    OptionParser optp(argc, argv);
    optp.option("--help", "-h", need_help, false, true);
    if( (need_help)||(argc == 1) ) {
        printHelp();
        return 0;
    }
    optp.option("--verbose", "-v", verbose, false, true);
    optp.option("--config", "-c", config_file, false);
    optp.option("--input", "-i", input_file, true);
    optp.option("--no-colors", "-nc", no_colors, false, true);
    if(!no_colors) {
        Log::inst().setColors();
    }
    ShapeletsConfig c;

    if(!config_file.empty()) {
        std::ifstream ifs(config_file);
        Stream(ifs, Stream::Text).readObject<ShapeletsConfig>(&c);
    }

    if(verbose) {
        Log::inst().setLogLevel(Log::DEBUG_LEVEL);
    }

    ifstream f(input_file);
    Stream s(f,Stream::Binary);

    Ptr<TimeSeries> ts(s.readObject<TimeSeries>());
    vector<Ptr<TimeSeries>> chopped = ts->chop();
    Dataset ds(chopped);
    ShapeletsAlgo alg(c);
    alg.run(ds);

    return 0;
}
