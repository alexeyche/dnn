
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
    auto opts = InitOptions(argc, argv, "AcousticModel");
    if (opts.NoLearning) {
        auto sim = BuildModel<
            TLayer<TSpikeSequenceNeuron, 256, TNeuronConfig<>>,
            TLayer<TSRMNeuron, 300, TNeuronConfig<TBasicSynapse, TLogExp>>
        >(opts);
        
        sim.Run();
    } else {
        auto sim = BuildModel<
            TLayer<TSpikeSequenceNeuron, 256, TNeuronConfig<>>,
            TLayer<TSRMNeuron, 300, TNeuronConfig<TBasicSynapse, TLogExp, TNoInput, TNearestStdp, TSumNorm, TMaxEntropyIP>>
        >(opts);

        if (opts.StatFile) {
            // sim.ListenBasicStats<2, 0>(0, std::numeric_limits<ui32>::max());
            // sim.ListenBasicStats<2, 1>(0, std::numeric_limits<ui32>::max());
            // sim.ListenBasicStats<2, 2>(0, std::numeric_limits<ui32>::max());

            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 2, 18>().Weight(); }, 0, 10000);
            // sim.ListenStat("M", [&]() { return sim.GetNeuron<1, 2>().ProbabilityModulation(); }, 0, 10000);
            // sim.ListenStat("Refr", [&]() { return sim.GetNeuron<1, 2>().State().VarRefr; }, 0, 10000);
            // sim.ListenStat("Adapt", [&]() { return sim.GetNeuron<1, 2>().State().VarAdapt; }, 0, 10000);
            
            sim.ListenStat("X", [&]() { return sim.GetLearningRule<1, 7>().State().X.Get(30); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Y", [&]() { return sim.GetLearningRule<1, 7>().State().Y; }, 0, std::numeric_limits<ui32>::max());
            
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 7, 30>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            
            // sim.ListenStat("FirstMoment", [&]() { return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(50); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("FirstMoment", [&]() { return sim.GetLearningRule<1, 1>().State().FirstMoment.Get(50); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("FirstMoment", [&]() { return sim.GetLearningRule<1, 2>().State().FirstMoment.Get(50); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("FirstMoment", [&]() { return sim.GetLearningRule<1, 3>().State().FirstMoment.Get(50); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("FirstMoment", [&]() { return sim.GetLearningRule<1, 4>().State().FirstMoment.Get(50); }, 0, std::numeric_limits<ui32>::max());
            
            // sim.ListenStat("SecondMoment", [&]() { return sim.GetLearningRule<1, 0>().State().SecondMoment.Get(50); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("SecondMoment", [&]() { return sim.GetLearningRule<1, 1>().State().SecondMoment.Get(50); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("SecondMoment", [&]() { return sim.GetLearningRule<1, 2>().State().SecondMoment.Get(50); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("SecondMoment", [&]() { return sim.GetLearningRule<1, 3>().State().SecondMoment.Get(50); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("SecondMoment", [&]() { return sim.GetLearningRule<1, 4>().State().SecondMoment.Get(50); }, 0, std::numeric_limits<ui32>::max());


            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 0, 50>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 1, 50>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 2, 50>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 3, 50>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 4, 50>().Weight(); }, 0, std::numeric_limits<ui32>::max());

            // sim.ListenStat("OptimalStdpB", [&]() { return sim.GetLearningRule<1, 0>().State().B; }, 0, 10000);   
            // sim.ListenStat("OptimalStdpC", [&]() { return sim.GetSynapse<1, 0, 11>().Potential(); }, 0, 10000);
            // sim.ListenStat("OptimalStdpC", [&]() { return sim.GetLearningRule<1, 0>().State().C.Get(11); }, 0, 10000);
            // sim.ListenStat("StdpX", [&]() { return sim.GetLearningRule<1, 0>().State().X.Get(0); }, 0, 10000);

            // sim.ListenStat("StdpY", [&]() { return sim.GetLearningRule<1, 0>().State().Y; }, 0, 10000);
            // sim.ListenStat("Ltd", [&]() { return sim.GetLearningRule<1, 0>().Norm().Ltd(sim.GetSynapse<1, 0, 10>().Weight()); }, 0, 10000);
        }

        sim.Run();
    }
    return 0;
}
