#include <iostream>

#include <dnn/base/entry.h>

#include <dnn/neuron/integrate_and_fire.h>
#include <dnn/synapse/hedonistic_synapse.h>
#include <dnn/protos/options.pb.h>
#include <dnn/receptive_field/ident.h>
#include <dnn/activation/determ.h>
#include <dnn/reinforcement/input_classifier.h>

using namespace NDnn;

int main(int argc, const char** argv) {
    auto opts = InitOptions(argc, argv, "HsModel");
    if (opts.NoLearning) {
        auto sim = BuildModel<
            TLayer<TIntegrateAndFire, 9, TNeuronConfig<THedonisticSynapse, TDeterm, TIdentReceptiveField, TNoLearning>>,
            TLayer<TIntegrateAndFire, 51, TNeuronConfig<THedonisticSynapse, TDeterm, TNoInput, TNoLearning>>,
            TLayer<TIntegrateAndFire, 2, TNeuronConfig<THedonisticSynapse, TDeterm, TNoInput, TNoLearning, TNoWeightNormalization, TInputClassifier>>
        >(opts);

        sim.Run();

    } else {
        auto sim = BuildModel<
            TLayer<TIntegrateAndFire, 9, TNeuronConfig<THedonisticSynapse, TDeterm, TIdentReceptiveField>>,
            TLayer<TIntegrateAndFire, 51, TNeuronConfig<THedonisticSynapse, TDeterm, TNoInput>>,
            TLayer<TIntegrateAndFire, 2, TNeuronConfig<THedonisticSynapse, TDeterm, TNoInput, TNoLearning, TNoWeightNormalization, TInputClassifier>>
        >(opts);

        if (opts.StatFile) {
            sim.ListenBasicStats<2, 0>(0, 2000);
            sim.ListenBasicStats<2, 1>(0, 2000);
            sim.CollectReward();
        }

        sim.Run();

    }
    return 0;
}