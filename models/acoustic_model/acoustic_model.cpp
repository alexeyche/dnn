
#include <iostream>

#include <dnn/base/entry.h>
#include <dnn/neuron/config.h>

#include <dnn/neuron/integrate_and_fire.h>
#include <dnn/neuron/spike_sequence_neuron.h>
#include <dnn/synapse/synapse.h>
#include <dnn/synapse/basic_synapse.h>
#include <dnn/synapse/stp_synapse.h>
#include <dnn/protos/options.pb.h>
#include <dnn/receptive_field/gauss.h>
#include <dnn/receptive_field/linear.h>
#include <dnn/receptive_field/ident.h>
#include <dnn/learning_rule/stdp.h>
#include <dnn/learning_rule/pre_post_stdp.h>
#include <dnn/activation/sigmoid.h>
#include <dnn/activation/determ.h>
#include <dnn/weight_normalization/multiplicative_norm.h>
#include <dnn/weight_normalization/min_max_norm.h>
#include <dnn/weight_normalization/sliding_ltd.h>
#include <dnn/weight_normalization/unit_norm.h>
#include <dnn/weight_normalization/nnmf_homeostasis.h>

using namespace NDnn;

int main(int argc, const char** argv) {
    auto opts = InitOptions(argc, argv, "AcousticModel");
    if (opts.NoLearning) {
        auto sim = BuildModel<
            TLayer<TSpikeSequenceNeuron, 256, TNeuronConfig<>>,
            TLayer<TIntegrateAndFire, 1, TNeuronConfig<TBasicSynapse, TDeterm>>
        >(opts);

        sim.Run();
    } else {
        auto sim = BuildModel<
            TLayer<TSpikeSequenceNeuron, 256, TNeuronConfig<>>,
            TLayer<TIntegrateAndFire, 1, TNeuronConfig<TBasicSynapse, TDeterm, TNoInput, TStdp, TSlidingLtd>>
        >(opts);

        if (opts.StatFile) {
            sim.ListenBasicStats<0, 46>(0, 10000);
            sim.ListenBasicStats<1, 0>(0, 10000);

            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 0, 1>().Weight(); }, 0, 10000);
            sim.ListenStat("StdpX", [&]() { return sim.GetLearningRule<1, 0>().State().X.Get(0); }, 0, 10000);

            sim.ListenStat("StdpY", [&]() { return sim.GetLearningRule<1, 0>().State().Y; }, 0, 10000);
        }

        sim.Run();
    }
    return 0;
}
