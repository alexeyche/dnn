
#include "kernel.h"
#include "kernel_factory.h"

#include <spikework/kernels/epsp.h>
#include <spikework/kernels/dot.h>
#include <spikework/kernels/entropy.h>
#include <spikework/kernels/rbf_dot.h>
#include <spikework/kernels/anova_dot.h>
#include <spikework/kernels/shoe.h>


namespace dnn {

KernelFactory::kernels_map_type KernelFactory::kernel_map;
KernelFactory::proc_map_type KernelFactory::kernel_proc_map;


KernelFactory::KernelFactory() {
    kernel_proc_map["Epsp"] = &createKernel<IKernelPreprocessor, EpspKernel>;

    kernel_map["Dot"] = &createKernel<IKernel, DotKernel>;
    kernel_map["Entropy"] = &createKernel<IKernel, EntropyKernel>;
    kernel_map["RbfDot"] = &createKernel<IKernel, RbfDotKernel>;
    kernel_map["AnovaDot"] = &createKernel<IKernel, AnovaDotKernel>;
    kernel_map["Shoe"] = &createKernel<IKernel, ShoeKernel>;
}

KernelFactory& KernelFactory::inst() {
    static KernelFactory _inst;
    return _inst;
}

Ptr<IKernel> KernelFactory::createKernel(string spec) {
    Ptr<IKernel> k;
    create_kernel(spec, kernel_map, k);
    return k;
}

Ptr<IKernelPreprocessor> KernelFactory::createKernelPreprocessor(string spec) {
    Ptr<IKernelPreprocessor> k;
    create_kernel(spec, kernel_proc_map, k);
    return k;
}



}
