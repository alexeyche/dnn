
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
#include <dnn/weight_normalization/rate_norm.h>
#include <dnn/intrinsic_plasticity/max_entropy.h>

using namespace NDnn;



int main(int argc, const char** argv) {
    auto opts = InitOptions(argc, argv, "FeedbackStdpModel");
    if (opts.NoLearning) {
        auto sim = BuildModel<
            TLayer<TSpikeSequenceNeuron, 100, TNeuronConfig<>>,
            TLayer<TIntegrateAndFire, 1000, TNeuronConfig<TBasicSynapse, TDeterm>>,
            TLayer<TIntegrateAndFire, 10, TNeuronConfig<TBasicSynapse, TDeterm>>
        >(opts);

        sim.Run();
    } else {
        auto sim = BuildModel<
            TLayer<TSpikeSequenceNeuron, 100, TNeuronConfig<>>,
            TLayer<TIntegrateAndFire, 1000, TNeuronConfig<TBasicSynapse, TDeterm, TNoInput, TResumeHiddenRule, TMinMaxNorm, TNoIntrinsicPlasticity, TNoReinforcement>>,
            TLayer<TIntegrateAndFire, 10, TNeuronConfig<TBasicSynapse, TDeterm, TNoInput, TResumeRule, TMinMaxNorm, TNoIntrinsicPlasticity, TNoReinforcement>>
        >(opts);

        constexpr auto layer_size = sim.LayersSize();

        for (auto& n: sim.GetMutLayer<layer_size-1>()) {
            n.GetMutLearningRule().SetTarget((*opts.TargetSpikes)[n.GetSpaceInfo().LocalId]);
        }


        if (opts.StatFile) {
            
            sim.ListenStat("Error", [&]() { return TGlobalCtx::Inst().GetMeanError(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Potential", [&]() { return sim.GetSynapse<2, 0, 0>().Potential(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Potential", [&]() { return sim.GetSynapse<2, 0, 1>().Potential(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Potential", [&]() { return sim.GetSynapse<2, 0, 2>().Potential(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<2, 0, 0>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<2, 0, 1>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<2, 0, 2>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<2, 0, 3>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            
            sim.ListenStat("Error", [&]() { return TGlobalCtx::Inst().GetMeanError(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Potential", [&]() { return sim.GetSynapse<2, 1, 0>().Potential(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Potential", [&]() { return sim.GetSynapse<2, 1, 1>().Potential(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Potential", [&]() { return sim.GetSynapse<2, 1, 2>().Potential(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<2, 1, 0>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<2, 1, 1>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<2, 1, 2>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            sim.ListenStat("Weight", [&]() { return sim.GetSynapse<2, 1, 3>().Weight(); }, 0, std::numeric_limits<ui32>::max());


            // sim.ListenStat("Error", [&]() { return sim.GetLearningRule<2, 0>().GetCurrentError(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Weight", [&]() { return sim.GetSynapse<2, 0, 0>().Weight(); }, 0, std::numeric_limits<ui32>::max());
            // sim.ListenStat("Synapse", [&]() { return sim.GetSynapse<2, 0, 0>().Potential(); }, 0, std::numeric_limits<ui32>::max());
        }

        sim.Run();

        double mean_error = 0;
        const auto& errors = TGlobalCtx::Inst().GetCumulativeError();
        for (const auto& n: sim.GetLayer<layer_size-1>()) {
            mean_error += errors[n.GetSpaceInfo().GlobalId]/sim.GetDuration();
        }

        mean_error = mean_error/sim.GetLayer<layer_size-1>().Size();
        L_INFO << "Mean error: " << mean_error;
    }
    return 0;
}
