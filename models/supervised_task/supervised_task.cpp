
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
#include <dnn/activation/exp.h>
#include <dnn/activation/determ.h>
#include <dnn/weight_normalization/sum_norm.h>
#include <dnn/weight_normalization/min_max_norm.h>

using namespace NDnn;



int main(int argc, const char** argv) {
    auto opts = InitOptions(argc, argv, "SupervisedTask");
    if (opts.NoLearning) {
        auto sim = BuildModel<
            TLayer<TSpikeSequenceNeuron, 15, TNeuronConfig<>>,
            TLayer<TSRMNeuron, 15, TNeuronConfig<TBasicSynapse, TLogExp>>,
            TLayer<TSRMNeuron, 15, TNeuronConfig<TBasicSynapse, TLogExp>>
        >(opts);

        sim.Run();
    } else {
        auto sim = BuildModel<
            TLayer<TSpikeSequenceNeuron, 15, TNeuronConfig<>>,
            TLayer<TSRMNeuron, 15, TNeuronConfig<TBasicSynapse, TLogExp, TNoInput, TSupervisedSpike, TMinMaxNorm>>,
            TLayer<TSRMNeuron, 15, TNeuronConfig<TBasicSynapse, TLogExp, TNoInput, TSupervisedSpike, TMinMaxNorm>>
        >(opts);
        
        ENSURE(opts.TargetSpikes, "Need target spikes");
        ENSURE(opts.TargetSpikes->Dim() == sim.GetLayer<1>().Size(), "Need target spikes with size of last layer: " << sim.GetLayer<1>().Size());

        for (auto& n: sim.GetMutLayer<2>()) {
            n.GetMutLearningRule().SetTarget((*opts.TargetSpikes)[n.GetSpaceInfo().LocalId]);
        }

        if (opts.StatFile) {
            sim.ListenBasicStats<1, 0>(0, std::numeric_limits<ui32>::max());

            // sim.ListenStat("Dw", [&]() { 
            //     return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(0)/std::sqrt(sim.GetLearningRule<1, 0>().State().SecondMoment.Get(0)+1e-08); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Dw", [&]() { 
            //     return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(1)/std::sqrt(sim.GetLearningRule<1, 0>().State().SecondMoment.Get(1)+1e-08); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Dw", [&]() { 
            //     return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(2)/std::sqrt(sim.GetLearningRule<1, 0>().State().SecondMoment.Get(2)+1e-08); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Dw", [&]() { 
            //     return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(3)/std::sqrt(sim.GetLearningRule<1, 0>().State().SecondMoment.Get(3)+1e-08); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Dw", [&]() { 
            //     return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(4)/std::sqrt(sim.GetLearningRule<1, 0>().State().SecondMoment.Get(4)+1e-08); }, 0, std::numeric_limits<ui32>::max());
            
            sim.ListenStat("Dw", [&]() { 
                return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(0); }, 0, sim.GetDuration());
            sim.ListenStat("Dw", [&]() { 
                return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(1); }, 0, sim.GetDuration());
            sim.ListenStat("Dw", [&]() { 
                return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(2); }, 0, sim.GetDuration());
            sim.ListenStat("Dw", [&]() { 
                return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(3); }, 0, sim.GetDuration());
            sim.ListenStat("Dw", [&]() { 
                return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(4); }, 0, sim.GetDuration());
            sim.ListenStat("Dw", [&]() { 
                return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(5); }, 0, sim.GetDuration());
            sim.ListenStat("Dw", [&]() { 
                return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(6); }, 0, sim.GetDuration());
            sim.ListenStat("Dw", [&]() { 
                return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(7); }, 0, sim.GetDuration());
            sim.ListenStat("Dw", [&]() { 
                return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(8); }, 0, sim.GetDuration());
            sim.ListenStat("Dw", [&]() { 
                return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(9); }, 0, sim.GetDuration());
            sim.ListenStat("Dw", [&]() { 
                return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(10); }, 0, sim.GetDuration());
            sim.ListenStat("Dw", [&]() { 
                return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(11); }, 0, sim.GetDuration());
            sim.ListenStat("Dw", [&]() { 
                return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(12); }, 0, sim.GetDuration());
            sim.ListenStat("Dw", [&]() { 
                return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(13); }, 0, sim.GetDuration());
            sim.ListenStat("Dw", [&]() { 
                return sim.GetLearningRule<1, 0>().State().FirstMoment.Get(14); }, 0, sim.GetDuration());
               
        }

        sim.Run();
    }
    return 0;
}
