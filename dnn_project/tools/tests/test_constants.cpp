
#include <dnn/base/constants.h>

using namespace dnn;

int main(int argc, char **argv) {
    vector<string> s;
    for(size_t i=1; i<argc; ++i) {
        s.push_back(argv[i]);
    }
    Constants c(s);
    cout << c;

}