


#include <dnn/core.h>
#include <dnn/util/option_parser.h>
#include <dnn/io/stream.h>
#include <dnn/util/time_series.h>

using namespace dnn;

const char * fft_usage = R"USAGE(SPIKEWORK ===========================================================
    fft subprogram for calculating fast fourier transform

        --input,  -i   input protobin 
        --output, -o  output protobin
        --help, -h
)USAGE";

int fft_sub(int argc, char **argv) {
    string input_file;
    string output_file;
    bool need_help;

    OptionParser optp(argc, argv);
    optp.option("--input", "-i", input_file, false);
    optp.option("--help", "-h", need_help, false, true);
    optp.option("--output", "-o", output_file, false);

    if((need_help)||(argc == 1)) {
        cout << fft_usage;
        return 0;
    }
    if(input_file.empty()) {
        throw dnnException() << "Need input file to work\n";
    }
    if(output_file.empty()) {
        throw dnnException() << "Need output file to work\n";
    }

    ifstream ff(input_file);
    Stream s(ff, Stream::Binary);
    TimeSeries* ts = s.safeReadObject<TimeSeries>();
    if(ts) {
        int *dims = new int[ts->data.size()];
        for(size_t i=0; i<ts->data.size(); ++i) {
            dims[i] = ts->data[i].values.size();
        }
        //kiss_fftndr_alloc
        delete []dims;
    }
    

    return 0;
}
