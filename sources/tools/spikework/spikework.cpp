

#include "fft.cpp"

const char * usage = R"USAGE(Tool for working with spikes and time series. Tool has subprograms
    fft
)USAGE";


int main(int argc, char **argv) {
    if(argc == 1) {
        cout << usage;
        return 0; 
    }
    if(strcmp(argv[1], "fft") == 0) {
        return fft_sub(--argc, ++argv);
    } else
    if((strcmp(argv[1], "-h") == 0)||(strcmp(argv[1], "--help") == 0)) {
       cout << usage; 
    } else {
       throw dnnException() << "Can't find subprogramm with name " << argv[1]; 
    }
    return 0;    
}
