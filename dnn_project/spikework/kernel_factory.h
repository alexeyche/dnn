#pragma once

#include <dnn/util/ptr.h>

namespace dnn {

class IKernel;
class IKernelPreprocessor;

class KernelFactory {
    template <typename BASE, typename INST> static Ptr<BASE> createKernel() { return new INST; }
public:
    typedef map<string, Ptr<IKernel> (*)()> kernels_map_type;
    typedef map<string, Ptr<IKernelPreprocessor> (*)()> proc_map_type;

    KernelFactory();
    static KernelFactory& inst();

    Ptr<IKernel> createKernel(string spec);
    Ptr<IKernelPreprocessor> createKernelPreprocessor(string spec);

    const kernels_map_type& getKernelsMap() {
        return kernel_map;
    }

    const proc_map_type& getKernelPreprocessorsMap() {
        return kernel_proc_map;
    }

private:
    template <typename MAP_TYPE, typename KERNEL_TYPE>
    void create_kernel(string spec, MAP_TYPE& m, KERNEL_TYPE &k) {
        size_t par_pos = spec.find_first_of("(");
        string spec_name = spec;
        if(string::npos != par_pos) {
            spec_name = spec.substr(0, par_pos);
        } else {
            spec = "";
        }
        auto k_ptr = m.find(trimC(spec_name));
        if(k_ptr == m.end()) {
            throw dnnException() << "Can't find kernel with specification " << spec_name;
        }
        k = k_ptr->second();
        k->processSpec(spec);
    }


    static kernels_map_type kernel_map;
    static proc_map_type kernel_proc_map;
};



}