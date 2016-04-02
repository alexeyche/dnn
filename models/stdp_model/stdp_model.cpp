
#include <iostream>

#include <dnn/base/entry.h>

#include <dnn/neuron/integrate_and_fire.h>
#include <dnn/neuron/spike_sequence_neuron.h>
#include <dnn/synapse/synapse.h>
#include <dnn/synapse/stp_synapse.h>
#include <dnn/protos/options.pb.h>
#include <dnn/receptive_field/gauss.h>
#include <dnn/learning_rule/stdp.h>
#include <dnn/activation/sigmoid.h>
#include <dnn/activation/determ.h>
#include <dnn/weight_normalization/nonlinear_norm.h>
#include <dnn/weight_normalization/min_max_norm.h>
#include <dnn/weight_normalization/soft_bounds.h>

using namespace NDnn;

int main(int argc, const char** argv) {
    auto opts = InitOptions(argc, argv, "StdpModel");
    if (opts.NoLearning) {
        auto sim = BuildModel<
            TLayer<TIntegrateAndFire, 100, TNeuronConfig<TBasicSynapse, TDeterm, TGaussReceptiveField>>,
            TLayer<TIntegrateAndFire, 100, TNeuronConfig<TSTPSynapse, TSigmoid>>
        >(opts);

        sim.Run();
    } else {
        auto sim = BuildModel<
            TLayer<TIntegrateAndFire, 100, TNeuronConfig<TBasicSynapse, TDeterm, TGaussReceptiveField>>,
            TLayer<TIntegrateAndFire, 100, TNeuronConfig<TBasicSynapse, TSigmoid, TNoInput, TStdp, TMinMaxNorm>>
        >(opts);

        if (opts.StatFile) {
            sim.ListenBasicStats<0, 55>(0, 1000);
            sim.ListenBasicStats<1, 55>(0, 1000);
            // sim.ListenStat("r", [&]() { return sim.GetSynapse<1, 55, 0>().State().r; }, 0, 1000);
            // sim.ListenStat("p", [&]() { return sim.GetSynapse<1, 55, 0>().State().p; }, 0, 1000);
        }
        
        sim.Run();
    }
    return 0;
}
