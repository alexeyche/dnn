
#include <iostream>

#include <dnn/base/entry.h>

#include <dnn/neuron/integrate_and_fire.h>
#include <dnn/neuron/spike_sequence_neuron.h>
#include <dnn/synapse/synapse.h>
#include <dnn/synapse/stp_synapse.h>
#include <dnn/protos/options.pb.h>
#include <dnn/receptive_field/gauss.h>
#include <dnn/learning_rule/stdp.h>
#include <dnn/learning_rule/pre_post_stdp.h>
#include <dnn/activation/sigmoid.h>
#include <dnn/activation/determ.h>
#include <dnn/weight_normalization/multiplicative_norm.h>
#include <dnn/weight_normalization/min_max_norm.h>
#include <dnn/weight_normalization/sliding_ltd.h>

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
            TLayer<TIntegrateAndFire, 100, TNeuronConfig<TSTPSynapse, TSigmoid, TNoInput, TStdp, TSlidingLtd>>
        >(opts);

        if (opts.StatFile) {
            // sim.ListenBasicStats<0, 55>(10000, 11000);
            sim.ListenBasicStats<1, 5>(10000, 11000);
            
            
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 5, 1>().Weight(); }, 10000, 11000);
            sim.ListenStat("StdpX", [&]() { return sim.GetLearningRule<1, 5>().State().X.Get(0); }, 10000, 11000);
            
            sim.ListenStat("StdpY", [&]() { return sim.GetLearningRule<1, 5>().State().Y; }, 10000, 11000);
            
        }
        
        sim.Run();
    }
    return 0;
}
