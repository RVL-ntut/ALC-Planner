#include "alc_planner/reward_evaluator.hpp"

#include <algorithm>
#include <cmath>

namespace alc_planner
{

namespace
{

float expDistanceDecay(const float l_minus, const float cl) {
    return std::exp(-(l_minus * l_minus) / (cl * cl));
}

float calibratedPLCIntrinsic(const SaliencyState& saliency_state,
                             const int keyframe_ix) {
    if (keyframe_ix < 0 ||
        keyframe_ix >= static_cast<int>(saliency_state.keyframes.size())) {
        return 0.0f;
    }

    return std::clamp(
        saliency_state.keyframes[static_cast<std::size_t>(keyframe_ix)]
                .plc_intrinsic *
            saliency_state.plc_calibration,
        0.0f, 1.0f);
}

float uncertaintyGainUB(const ALCCandidate& candidate,
                        const bool use_variance_uncertainty) {
    if (use_variance_uncertainty) {
        return candidate.graph_dist_var;
    }
    return candidate.graph_dist - candidate.euclidean_dist;
}

float uncertaintyGainExact(const ALCCandidate& candidate,
                           const bool use_variance_uncertainty) {
    if (use_variance_uncertainty) {
        return candidate.graph_dist_var;
    }
    return candidate.graph_dist - candidate.map_dist;
}

}  // namespace

RewardEvaluator::RewardEvaluator(Params params) : params_(params) {}

void RewardEvaluator::fillRewardUB(ALCCandidate& candidate,
                                   const SaliencyState& saliency_state) const {
    candidate.delta_U_ub =
        uncertaintyGainUB(candidate, params_.use_variance_uncertainty);
    const float l_minus_ub = candidate.graph_dist + candidate.euclidean_dist;
    const float exp_factor = expDistanceDecay(l_minus_ub, params_.cl);

    float plc_sum = 0.0f;
    for (const int keyframe_ix : candidate.keyframe_ixs) {
        plc_sum +=
            calibratedPLCIntrinsic(saliency_state, keyframe_ix) * exp_factor;
    }

    candidate.P_lc_ub = std::min(1.0f, plc_sum);
    candidate.reward_ub = -params_.ct * candidate.euclidean_dist +
                          candidate.P_lc_ub * candidate.delta_U_ub;
}

void RewardEvaluator::fillReward(ALCCandidate& candidate,
                                 const SaliencyState& saliency_state) const {
    candidate.delta_U =
        uncertaintyGainExact(candidate, params_.use_variance_uncertainty);
    const float l_minus = candidate.graph_dist + candidate.map_dist;

    float prob_all_fail = 1.0f;
    for (const int keyframe_ix : candidate.keyframe_ixs) {
        const float plc_i =
            calibratedPLCIntrinsic(saliency_state, keyframe_ix) *
            expDistanceDecay(l_minus, params_.cl);
        prob_all_fail *= (1.0f - plc_i);
    }

    candidate.P_lc = 1.0f - prob_all_fail;
    candidate.reward =
        -params_.ct * candidate.map_dist + candidate.P_lc * candidate.delta_U;
}

}  // namespace alc_planner
