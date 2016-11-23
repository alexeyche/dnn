
#include <iostream>

#include <dnn/base/entry.h>
#include <dnn/neuron/config.h>

#include <dnn/neuron/srm_neuron.h>
#include <dnn/synapse/basic_synapse.h>
#include <dnn/synapse/stp_synapse.h>
#include <dnn/protos/options.pb.h>
#include <dnn/receptive_field/gauss.h>
#include <dnn/receptive_field/linear.h>
#include <dnn/receptive_field/ident.h>
#include <dnn/reinforcement/sequence_likelihood.h>
#include <dnn/learning_rule/sequence_learning.h>
#include <dnn/learning_rule/resume_rule.h>
#include <dnn/learning_rule/resume_hidden_rule.h>
#include <dnn/learning_rule/optimal_stdp.h>
#include <dnn/learning_rule/supervised_spike.h>
#include <dnn/activation/exp.h>
#include <dnn/activation/sigmoid.h>

using namespace NDnn;


int main(int argc, const char** argv) {
    auto opts = InitOptions(argc, argv, "SeqModel");
    if (opts.NoLearning) {
        auto sim = BuildModel<
            TLayer<TSRMNeuron, 100, TNeuronConfig<TBasicSynapse, TSigmoid>>,
            TLayer<TSRMNeuron, 10, TNeuronConfig<TBasicSynapse, TSigmoid>>
        >(opts);

        sim.Run();
    } else {
        auto sim = BuildModel<
            TLayer<TSRMNeuron, 100, TNeuronConfig<TBasicSynapse, TSigmoid, TNoInput, TSequenceLearning, TNoWeightNormalization, TNoIntrinsicPlasticity, TNoReinforcement>>,
            TLayer<TSRMNeuron, 10, TNeuronConfig<TBasicSynapse, TSigmoid, TNoInput, TSequenceLearning, TNoWeightNormalization, TNoIntrinsicPlasticity, TSequenceLikelihood>>
        >(opts);

        constexpr auto layer_size = sim.LayersSize();

        for (auto& n: sim.GetMutLayer<layer_size-1>()) {
            n.GetMutReinforcement().SetTarget((*opts.TargetSpikes)[n.GetSpaceInfo().LocalId]);
        }


        if (opts.StatFile) {
            sim.CollectReward();
        }

        sim.Run();

        // double mean_error = 0;
        // const auto& errors = TGlobalCtx::Inst().GetCumulativeError();
        // for (const auto& n: sim.GetLayer<layer_size-1>()) {
        //     mean_error += errors[n.GetSpaceInfo().GlobalId]/sim.GetDuration();
        // }

        // mean_error = mean_error/sim.GetLayer<layer_size-1>().Size();
        // L_INFO << "Mean error: " << mean_error;
    }
    return 0;
}
