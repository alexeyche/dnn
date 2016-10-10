
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
#include <dnn/learning_rule/resume_rule.h>
#include <dnn/learning_rule/resume_hidden_rule.h>
#include <dnn/learning_rule/optimal_stdp.h>
#include <dnn/learning_rule/supervised_spike.h>
#include <dnn/activation/exp.h>
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
    auto opts = InitOptions(argc, argv, "FeedbackStdpModel");
    if (opts.NoLearning) {
        auto sim = BuildModel<
            TLayer<TSpikeSequenceNeuron, 100, TNeuronConfig<>>,
            TLayer<TIntegrateAndFire, 100, TNeuronConfig<TBasicSynapse, TDeterm>>,
            TLayer<TIntegrateAndFire, 10, TNeuronConfig<TBasicSynapse, TDeterm>>
        >(opts);

        sim.Run();
    } else {
        auto sim = BuildModel<
            TLayer<TSpikeSequenceNeuron, 100, TNeuronConfig<>>,
            TLayer<TIntegrateAndFire, 100, TNeuronConfig<TBasicSynapse, TDeterm, TNoInput, TResumeHiddenRule, TMinMaxNorm, TNoIntrinsicPlasticity, TNoReinforcement>>,
            TLayer<TIntegrateAndFire, 10, TNeuronConfig<TBasicSynapse, TDeterm, TNoInput, TResumeRule, TMinMaxNorm, TNoIntrinsicPlasticity, TNoReinforcement>>
        >(opts);

        // for (auto& n: sim.GetMutLayer<2>()) {
        //     n.GetMutCostFunction().SetTarget(opts.TargetTimeSeries->GetVector(n.GetSpaceInfo().LocalId));
        // }
        for (auto& n: sim.GetMutLayer<2>()) {
            n.GetMutLearningRule().SetTarget((*opts.TargetSpikes)[n.GetSpaceInfo().LocalId]);
        }


        if (opts.StatFile) {
            sim.ListenStat("Error", [&]() { return sim.GetLearningRule<1, 0>().GetCurrentError(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Error", [&]() { return sim.GetLearningRule<1, 1>().GetCurrentError(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Error", [&]() { return sim.GetLearningRule<1, 2>().GetCurrentError(); }, 0, std::numeric_limits<ui32>::max());

            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 0, 0>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 0, 3>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 0, 4>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 0, 5>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 0, 6>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 0, 7>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 0, 8>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 0, 9>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<1, 0, 10>().Weight(); }, 0, std::numeric_limits<ui32>::max());


            // sim.ListenStat("Error", [&]() { return sim.GetLearningRule<2, 0>().GetCurrentError(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<2, 0, 0>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Synapse", [&]() { return sim.GetSynapse<2, 0, 0>().Potential(); }, 0, std::numeric_limits<ui32>::max());
        }

        sim.Run();

        double mean_error = 0;
        const auto& errors = TGlobalCtx::Inst().GetCumulativeError();

        for (const auto& error: errors) {
            mean_error += error/sim.GetDuration();
        }
        mean_error = mean_error/sim.GetLayer<2>().Size();
        L_INFO << "Mean error: " << mean_error;
    }
    return 0;
}
