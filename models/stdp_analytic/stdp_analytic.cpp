
#include <iostream>

#include <dnn/base/entry.h>
#include <dnn/neuron/config.h>

#include <dnn/neuron/integrate_and_fire.h>
#include <dnn/neuron/srm_neuron.h>
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
#include <dnn/learning_rule/optimal_stdp.h>
#include <dnn/activation/sigmoid.h>
#include <dnn/activation/log_exp.h>
#include <dnn/activation/determ.h>
#include <dnn/weight_normalization/multiplicative_norm.h>
#include <dnn/weight_normalization/nonlinear_norm.h>
#include <dnn/weight_normalization/min_max_norm.h>
#include <dnn/weight_normalization/sliding_ltd.h>
#include <dnn/weight_normalization/unit_norm.h>
#include <dnn/weight_normalization/log_norm.h>
#include <dnn/weight_normalization/sum_norm.h>
#include <dnn/intrinsic_plasticity/max_entropy.h>

using namespace NDnn;



int main(int argc, const char** argv) {
    auto opts = InitOptions(argc, argv, "StdpAnalytic");
    if (opts.NoLearning) {
        auto sim = BuildModel<
            TLayer<TSpikeSequenceNeuron, 256, TNeuronConfig<>>,
            TLayer<TSRMNeuron, 10, TNeuronConfig<TBasicSynapse, TLogExp>>
        >(opts);

        sim.Run();
    } else {
        auto sim = BuildModel<
            TLayer<TSpikeSequenceNeuron, 256, TNeuronConfig<>>,
            TLayer<TSRMNeuron, 10, TNeuronConfig<TBasicSynapse, TLogExp, TNoInput, TNearestStdp, TSumNorm, TMaxEntropyIP>>
        >(opts);

        if (opts.StatFile) {
            // sim.ListenStat("Momentum", [&]() { return sim.GetLearningRule<1, 0>().State().Momentum.Get(256); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Momentum", [&]() { return sim.GetLearningRule<1, 1>().State().Momentum.Get(256); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Momentum", [&]() { return sim.GetLearningRule<1, 2>().State().Momentum.Get(256); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Momentum", [&]() { return sim.GetLearningRule<1, 3>().State().Momentum.Get(256); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Momentum", [&]() { return sim.GetLearningRule<1, 4>().State().Momentum.Get(256); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Momentum", [&]() { return sim.GetLearningRule<1, 5>().State().Momentum.Get(256); }, 0, std::numeric_limits<ui32>::max());

            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 0, 256>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 1, 256>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 2, 256>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 3, 256>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 4, 256>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 5, 256>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 6, 256>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 7, 256>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 8, 256>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 9, 256>().Weight(); }, 0, std::numeric_limits<ui32>::max());
        }

        sim.Run();
    }
    return 0;
}
