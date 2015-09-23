
#include <iostream>
#include <stdlib.h>
#include <fstream>

#include <dnn/core.h>
#include <mpl/mpl.h>
#include <dnn/util/option_parser.h>
#include <dnn/io/stream.h>
#include <dnn/util/time_series.h>
#include <dnn/base/factory.h>
#include <dnn/util/log/log.h>

using namespace dnn;
using namespace mpl;
using namespace rapidjson;

const char * usage = R"USAGE(
Matching Pursuit Learning tool

    --input, i        input time series
    --filter, -f      filter protobin file
    --spikes, -s      spikes to save
    --matches, -m     filter matches to save
    --dim, -d         index of dimension of input time series to use
    --restored, -r    calculate restored time series and print error (not required)
    --verbose, -v     turn on verbose logging
    --no-colors, -nc  no colors while logging
    --help, -h        for this menu
    --config, -c      json file with structure like this, to override defaults:
%s
)USAGE";

void printHelp() {
    MatchingPursuitConfig c;
    ostringstream s;
    Stream(s, Stream::Text).writeObject(&c);
    string json_str = s.str();
    printf(usage, json_str.c_str());
}

int main(int argc, char **argv) {

    if(argc == 1) {
        printHelp();
        return 0;
    }
    OptionParser optp(argc, argv);

    string input_file;
    string spikes_file;
    string config_file;
    string filter_file;
    string matches_file;
    bool need_help = false;
    bool verbose = false;
    int dimension = 0;
    bool no_colors = false;
    string restored_ts;
    optp.option("--help", "-h", need_help, false, true);
    if(need_help) {
        printHelp();
        return 0;
    }
    optp.option("--input", "-i", input_file, true);
    optp.option("--filter", "-f", filter_file);
    optp.option("--spikes", "-s", spikes_file, false);
    optp.option("--matches", "-m", matches_file, false);
    optp.option("--config", "-c", config_file, false);
    optp.option("--dim", "-d", dimension, false);
    optp.option("--restored", "-r", restored_ts, false);
    optp.option("--verbose", "-v", verbose, false, true);
    optp.option("--no-colors", "-nc", no_colors, false, true);

    if(verbose) {
        Log::inst().setLogLevel(Log::DEBUG_LEVEL);
    }
    if(!no_colors) {
        Log::inst().setColors();
    }

    MatchingPursuitConfig c;

    if(!config_file.empty()) {
        std::ifstream ifs(config_file);
        c = Stream(ifs, Stream::Text).readDynamic<MatchingPursuitConfig>().ref();
    }
    MatchingPursuit mpl(c);

    if( (fileExists(filter_file)) && ( (!c.learn) || c.continue_learning )) {
        L_INFO << "Reading filter from " << filter_file;
        std::ifstream ifs(filter_file);
        Ptr<DoubleMatrix> f = Stream(ifs, Stream::Binary).readDynamic<DoubleMatrix>();

        mpl.setFilter(f.ref());

        delete f.ptr();

    }

    std::ifstream ifs(input_file);
    Ptr<TimeSeries> ts = Stream(ifs, Stream::Binary).read<TimeSeries>();
    if(dimension>=ts->dim()) {
        throw dnnException() << "Can't find dimension with index " << dimension << " in input time series\n";
    }
    MatchingPursuit::MPLReturn r = mpl.run(*ts, dimension);
    if(c.learn) {
        std::ofstream ofs(filter_file);
        DoubleMatrix f = mpl.getFilter();
        Stream(ofs, Stream::Binary).writeObject(&f);
    }
    if(!spikes_file.empty()) {
        std::ofstream ofs(spikes_file);
        Stream s(ofs, Stream::Binary);
        Ptr<SpikesList> sl = MatchingPursuit::convertMatchesToSpikes(r.matches);
        s.writeObject(sl.ptr());
    }
    if(!matches_file.empty()) {
        std::ofstream ofs(matches_file);
        Stream s(ofs, Stream::Binary);
        for(auto &m: r.matches) {
            s.writeObject(&m);
        }
    }



    if(!restored_ts.empty()) {
        vector<double> v = mpl.restore(r.matches);
        double v_denom = 0.0;
        for(auto &v_el :v) {
            v_denom += v_el * v_el;
        }
        v_denom = sqrt(v_denom);

        TimeSeries ts_rest(v);
        std::ofstream s(restored_ts);
        Stream(s, Stream::Binary).writeObject(&ts_rest);
        double acc_error = 0;
        for(size_t vi=0; vi<ts->length(); ++vi) {
            const double& orig_val = ts->getValueAtDim(vi, dimension);
            double rest_val;
            if(vi < v.size()) {
                rest_val = v[vi]/v_denom;
            } else {
                rest_val = 0.0;
            }
            acc_error += (orig_val - rest_val)*(orig_val - rest_val);
        }
        L_INFO << "Accumulated error: " << 100000*acc_error/ts->length();
        // Document d;
        // d.SetObject();
        // d.AddMember("accum_error", acc_error, d.GetAllocator());
        // cout << Json::stringify(d) << "\n";
    }
    return 0;
}