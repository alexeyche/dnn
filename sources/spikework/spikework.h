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
    

    class Stack {
    public:
    	Ptr<SerializableBase> pop() {
            if(stack.size() == 0) {
                throw dnnException() << "stack is exhausted\n";
            }
    		Ptr<SerializableBase> b = stack.back();
    		stack.pop_back();
    		return b;
    	}
        Ptr<SerializableBase> back() {
            if(stack.size() == 0) {
                throw dnnException() << "stack is exhausted\n";
            }
            return stack.back();
        }
        void push(Ptr<SerializableBase> p) {
            stack.push_back(p);
        }
	private:    	
    	vector<Ptr<SerializableBase>> stack;        
    };

private:
    Stack s;
    vector<Ptr<Processor>> processors;
    processors_map_type proc_map;
};




}
