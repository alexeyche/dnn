
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
#include <dnn/learning_rule/supervised_spike.h>
#include <dnn/learning_rule/optimal_stdp.h>
#include <dnn/activation/sigmoid.h>
#include <dnn/activation/log_exp.h>
#include <dnn/activation/determ.h>
#include <dnn/weight_normalization/sum_norm.h>

using namespace NDnn;



int main(int argc, const char** argv) {
    auto opts = InitOptions(argc, argv, "SupervisedTask");
    if (opts.NoLearning) {
        auto sim = BuildModel<
            TLayer<TSpikeSequenceNeuron, 10, TNeuronConfig<>>,
            TLayer<TSRMNeuron, 10, TNeuronConfig<TBasicSynapse, TLogExp>>
        >(opts);

        sim.Run();
    } else {
        auto sim = BuildModel<
            TLayer<TSpikeSequenceNeuron, 10, TNeuronConfig<>>,
            TLayer<TSRMNeuron, 10, TNeuronConfig<TBasicSynapse, TLogExp, TNoInput, TSupervisedSpike, TSumNorm>>
        >(opts);
        
        for (auto& n: sim.GetMutLayer<1>()) {
            n.GetMutLearningRule().SetTarget((*opts.InputSpikes)[n.GetSpaceInfo().LocalId]);
        }

        if (opts.StatFile) {
            sim.ListenBasicStats<1, 0>(0, std::numeric_limits<ui32>::max());

            sim.ListenStat("FirstMoment", [&]() { return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(0); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("FirstMoment", [&]() { return sim.GetLearningRule<1, 1>().State().FirstMoment.Get(0); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("FirstMoment", [&]() { return sim.GetLearningRule<1, 2>().State().FirstMoment.Get(0); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("FirstMoment", [&]() { return sim.GetLearningRule<1, 3>().State().FirstMoment.Get(0); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("FirstMoment", [&]() { return sim.GetLearningRule<1, 4>().State().FirstMoment.Get(0); }, 0, std::numeric_limits<ui32>::max());

            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 0, 0>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 1, 0>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 2, 0>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 3, 0>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 4, 0>().Weight(); }, 0, std::numeric_limits<ui32>::max());
        }

        sim.Run();
    }
    return 0;
}
