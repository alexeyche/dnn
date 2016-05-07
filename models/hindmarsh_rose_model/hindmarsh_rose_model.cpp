#include <iostream>

#include <dnn/base/entry.h>
#include <dnn/neuron/config.h>

#include <dnn/neuron/hindmarsh_rose.h>
#include <dnn/neuron/spike_sequence_neuron.h>
#include <dnn/synapse/synapse.h>
#include <dnn/synapse/basic_synapse.h>
#include <dnn/protos/options.pb.h>
#include <dnn/receptive_field/gauss.h>
#include <dnn/receptive_field/ident.h>
#include <dnn/learning_rule/stdp.h>
#include <dnn/activation/sigmoid.h>

using namespace NDnn;

int main(int argc, const char** argv) {
    auto opts = InitOptions(argc, argv, "HindmarshRoseModel");
    if (opts.NoLearning) {
        auto sim = BuildModel<
            TLayer<TIntegrateAndFire, 100, TNeuronConfig<TBasicSynapse, TSigmoid, TGaussReceptiveField, TNoLearning>>,
            TLayer<THindmarshRose, 100, TNeuronConfig<TBasicSynapse, TSigmoid, TNoInput, TNoLearning>>
        >(opts);

        sim.Run();

    } else {
        auto sim = BuildModel<
            TLayer<TIntegrateAndFire, 100, TNeuronConfig<TBasicSynapse, TSigmoid, TGaussReceptiveField>>,
            TLayer<THindmarshRose, 100, TNeuronConfig<TBasicSynapse, TSigmoid, TNoInput, TStdp>>
        >(opts);

        if (opts.StatFile) {
            sim.ListenBasicStats<0, 55>(0, 1000);
            sim.ListenBasicStats<1, 55>(0, 1000);
            sim.ListenStat("StdpY", [&]() { return sim.GetLearningRule<1, 10>().State().Y; }, 0, 1000);
            sim.ListenStat("StdpX", [&]() { return sim.GetLearningRule<1, 10>().State().X.Get(10); }, 0, 1000);
            sim.ListenStat("Synapse", [&]() { return sim.GetSynapse<1, 10, 10>().Potential(); }, 0, 1000);
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 10, 10>().Weight(); }, 0, 1000);
        }

        sim.Run();

    }
    return 0;
}
