
#include "util.h"

#include <iomanip>
#include <execinfo.h>

namespace dnn {

vector<string> split_into(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split_into(s, delim, elems);
    return elems;
}

vector<string> splitBySubstr(const string &s_inp, const string &delimiter, const string &not_include) {
    string s(s_inp);
    vector<string> out;

    size_t pos = 0;
    std::string acc_token;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        // cout << "token: " << token << "\n";
        if((!not_include.empty()) && (token.find_first_of(not_include) != std::string::npos)) {
            if(!acc_token.empty()) {
                acc_token += delimiter;
            }
            acc_token += token;
            // cout << token << " " << acc_token << " 1\n";
        } else {
            if(!acc_token.empty()) {
                // cout << token << " " << acc_token << " 2\n";
                token = acc_token + delimiter + token;
                acc_token = std::string();
            }
            out.push_back(trimC(token));
        }
        s.erase(0, pos + delimiter.length());
    }
    if(!acc_token.empty()) {
        // cout << s << " " << acc_token << " 3\n";
        s = acc_token + delimiter + s;
    }
    out.push_back(trimC(s));
    return out;
}

void trim(string &str, string symbols) {
    size_t endpos = str.find_last_not_of(symbols);
    if( string::npos != endpos )
    {
        str = str.substr( 0, endpos+1 );
    }
    size_t startpos = str.find_first_not_of(symbols);
    if( string::npos != startpos )
    {
        str = str.substr( startpos );
    }
}

string trimC(const string &str, string symbols) {
    string s(str);
    trim(s, symbols);
    return s;
}
double normal_distr_var = -1;

double getUnif() {
    return (double)rand()/(double)RAND_MAX;
}


double getExp(double rate) {
    double u = getUnif();
    return -log(u)/rate;
}

double getNorm() {
    if(normal_distr_var<0) {
        double U = getUnif();
        double V = getUnif();
        normal_distr_var = sqrt(-2*log(U)) * cos(2*PI*V);
        return(sqrt(-2*log(U)) * sin(2*PI*V));
    } else {
        double ret = normal_distr_var;
        normal_distr_var=-1;
        return(ret);
    }
}

double getLogNorm(double logmean, double logsd) {
    return( exp(logmean + logsd*getNorm()) );
}

void swapInd(size_t *l, size_t *r) {
    size_t temp = *r;
    *r = *l;
    *l = temp;
}

double getUnifBetween(double low, double high) {
    return( low + getUnif()*(high-low) );
}

double sampleDelay(double gain, double rate) {
    if(fabs(rate) < 0.00001) return 0.0;
    return gain*getExp(rate);
}

string strip_white(const string& input)
{
   size_t b = input.find_first_not_of(' ');
   if (b == string::npos) b = 0;
   return input.substr(b, input.find_last_not_of(' ') + 1 - b);
}

string strip_comments(const string& input, const string& delimiters)
{
   return strip_white(input.substr(0, input.find_first_of(delimiters)));
}

long getFileSize(string filename) {
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

bool fileExists(const std::string& name) {
    ifstream f(name.c_str());
    if (f.good()) {
        f.close();
        return true;
    } else {
        f.close();
        return false;
    }
}

bool strStartsWith(const string &s, const string &prefix) {
    return s.substr(0, prefix.size()) == prefix;
}

bool strEndsWith(const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

vector<double> parseParenthesis(const string &s) {
    if((std::count(s.begin(), s.end(), '(') != 1)||(std::count(s.begin(), s.end(), ')') != 1)) {
        throw dnnException()<< "Bad string to parse parenthesis: " << s << "\n";
    }
    vector<string> left_side = split(s, '(');
    vector<string> right_side = split(left_side.back(), ')');
    vector<string> content = split(right_side.front(), ',');
    vector<double> out;
    for(auto it=content.begin(); it != content.end(); ++it) {
        out.push_back(stof(*it));
    }
    return out;
}


vector<IndexSlice> dispatchOnThreads(size_t elements_size, size_t jobs) {
    vector<IndexSlice> out;
    for(size_t ji=0; ji < jobs; ji++) {
        int el_per_thread;
        if(fabs(fmod(elements_size, jobs)) < 0.000001) {
            el_per_thread = elements_size / jobs;
        } else {
            el_per_thread = (elements_size + jobs - 1) / jobs;
        }

        size_t first = min( ji    * el_per_thread, elements_size );
        size_t last  = min( (ji+1) * el_per_thread, elements_size );
        if(first != last) {
            out.push_back( IndexSlice(first, last) );
        }
    }
    return out;
}
void replaceStr( string &s, const string &search, const string &replace, size_t num) {
    size_t repl_num = 0;
    for( size_t pos = 0; ; pos += replace.length() ) {
        if(repl_num>=num) break;
        // Locate the substring to replace
        pos = s.find( search, pos );
        if( pos == string::npos ) break;
        // Replace by erasing and inserting
        s.erase( pos, search.length() );
        s.insert( pos, replace );
        repl_num+=1;
    }
}

map<string, string> parseArgOptionsPairs(const vector<string> &opts) {
    map<string, string> opt_pairs;
    for(size_t i=0; i<opts.size(); i+=2) {
        if ((i+1) >= opts.size()) {
            throw dnnException()<< "Free option without an argument: " << opts[i] << "\n";
        }
        string optname = opts[i];
        string optvalue =  opts[i+1];

        size_t startpos = optname.find_first_not_of("--");
        if( string::npos != startpos ) {
            optname = optname.substr( startpos );
        } else {
            throw dnnException()<< "Free option must be like that: --free-option\n" << "\t got " << optname << "\n";
        }
        opt_pairs.insert(
            std::make_pair(
                "@" + optname,
                optvalue
            )
        );
    }
    return opt_pairs;
}


unsigned long upper_power_of_two(unsigned long v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}


ostream& printNow(ostream &o) {
    char buffer[23];
    time_t now = time(0);
    strftime(buffer, 23, "%Y-%m-%d %H:%M:%S ", localtime (&now));
    string s(buffer);
    o << s;
    return o;
}

void printBackTrace() {
    int j, nptrs;
    void *buffer[100];
    char **strings;

    nptrs = backtrace(buffer, 100);
    printf("backtrace() returned %d addresses\n", nptrs);

    /* The call backtrace_symbols_fd(buffer, nptrs, STDOUT_FILENO)
      would produce similar output to the following: */

    strings = backtrace_symbols(buffer, nptrs);
    if (strings == NULL) {
       perror("backtrace_symbols");
       exit(EXIT_FAILURE);
    }

    for (j = 0; j < nptrs; j++)
       printf("%s\n", strings[j]);

    free(strings);
}


double atomicDoubleAdd(atomic<double> &f, double d) {
    double old = f.load(memory_order_consume);
    double desired =  old+d;
    while (!f.compare_exchange_weak(
            old
          , desired
          , memory_order_release
          , memory_order_consume
    )) {
        desired = old + d;
    }
    return desired;
}

pair<size_t, size_t> parseConnectionSpec(string s) {
    vector<string> aff = splitBySubstr(s, "->");
    if (aff.size() != 2) {
        throw dnnException() << "connection specification not right: need 2 afferents separated by \"->\"\n";
    }

    return make_pair(stoi(aff[0]), stoi(aff[1]));
}


}