#pragma once

#include <dnn/util/ptr.h>
#include <dnn/io/serialize.h>


namespace dnn {


class Processor;

class Spikework {
    typedef map<string, Ptr<Processor> (*)()> processors_map_type;
    template<typename INST> static Ptr<Processor> createProcessor() { return new INST; }
public:
    Spikework(const vector<string> &args);
    ~Spikework();

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
        template<typename T>
        void push(Ptr<T> p) {
            stack.push_back(p. template as<SerializableBase>());
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
