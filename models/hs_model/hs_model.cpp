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
            TLayer<TIntegrateAndFire, 51, TNeuronConfig<TBasicSynapse, TDeterm, TNoInput, TNoLearning>>,
            TLayer<TIntegrateAndFire, 2, TNeuronConfig<TBasicSynapse, TDeterm, TNoInput, TNoLearning, TInputClassifier>>
        >(opts);

        sim.Run();

    } else {
        auto sim = BuildModel<
            TLayer<TIntegrateAndFire, 9, TNeuronConfig<TBasicSynapse, TDeterm, TIdentReceptiveField>>,
            TLayer<TIntegrateAndFire, 51, TNeuronConfig<TBasicSynapse, TDeterm, TNoInput>>,
            TLayer<TIntegrateAndFire, 2, TNeuronConfig<TBasicSynapse, TDeterm, TNoInput, TNoLearning, TInputClassifier>>
        >(opts);

        if (opts.StatFile) {
            sim.ListenBasicStats<2, 0>(0, 1000);
            sim.ListenBasicStats<2, 1>(0, 1000);
            sim.ListenStat("Synapse", [&]() { return sim.GetSynapse<2, 0, 1>().Potential(); }, 0, 1000);
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<2, 1, 1>().Weight(); }, 0, 1000);
        }

        sim.Run();

    }
    return 0;
}