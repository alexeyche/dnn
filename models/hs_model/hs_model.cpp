#include <iostream>

#include <dnn/base/entry.h>

#include <dnn/neuron/integrate_and_fire.h>
#include <dnn/synapse/basic_synapse.h>
#include <dnn/protos/options.pb.h>
#include <dnn/receptive_field/ident.h>
#include <dnn/activation/determ.h>
#include <dnn/reinforcement/input_classifier.h>

using namespace NDnn;

int main(int argc, const char** argv) {
    auto opts = InitOptions(argc, argv, "HsModel");
    if (opts.NoLearning) {
        auto sim = BuildModel<
            TLayer<TIntegrateAndFire, 9, TNeuronConfig<TBasicSynapse, TDeterm, TIdentReceptiveField, TNoLearning>>,
            TLayer<TIntegrateAndFire, 45, TNeuronConfig<TBasicSynapse, TDeterm, TNoInput, TNoLearning>>
        >(opts);

        sim.Run();

    } else {
        auto sim = BuildModel<
            TLayer<TIntegrateAndFire, 9, TNeuronConfig<TBasicSynapse, TDeterm, TIdentReceptiveField>>,
            TLayer<TIntegrateAndFire, 45, TNeuronConfig<TBasicSynapse, TDeterm, TNoInput>>
        >(opts);

        if (opts.StatFile) {
            sim.ListenBasicStats<0, 8>(0, 1000);
            sim.ListenBasicStats<1, 55>(0, 1000);
            sim.ListenStat("Synapse", [&]() { return sim.GetSynapse<1, 10, 10>().Potential(); }, 0, 1000);
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 10, 10>().Weight(); }, 0, 1000);
        }

        sim.Run();

    }
    return 0;
}