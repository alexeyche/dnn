#pragma once

#include <dnn/util/ptr.h>
#include <dnn/io/serialize.h>


namespace dnn {


class Processor;

class Spikework {
public:
    typedef map<string, Ptr<Processor> (*)()> processors_map_type;
    template<typename INST> static Ptr<Processor> createProcessor() { return new INST; }

    Spikework(const vector<string> &args);
    

    class Field {
    public:
    	Ptr<SerializableBase> pop_input() {
            if(inputs.size() == 0) {
                throw dnnException() << "input stack is exhausted\n";
            }
    		Ptr<SerializableBase> b = inputs.back();
    		inputs.pop_back();
    		return b;
    	}
        void push_input(Ptr<SerializableBase> inp) {
            inputs.push_back(inp);
        }
	private:    	
    	vector<Ptr<SerializableBase>> inputs;
    	vector<Ptr<SerializableBase>> outputs;
    };

private:
    Field f;
    vector<Ptr<Processor>> processors;
    processors_map_type proc_map;
};

class Processor {
public:
	void processDefaultArgs(const vector<string> &args);
    
    virtual void processArgs(const vector<string> &args) {};
	
    virtual void process(Spikework::Field &f) = 0;
    
    void start(Spikework::Field &f);
    void end(Spikework::Field &f) {}
private:
    string input_filename;
    string output_filename;
};



}
