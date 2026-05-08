#include "alc_planner/bnb_selector.hpp"

#include <algorithm>
#include <optional>

#include "alc_planner/map_utils.hpp"

namespace alc_planner
{

BNBSelector::BNBSelector(Params params)
    : params_(params), evaluator_(params_) {}

std::optional<ALCCandidate> BNBSelector::select(
    std::vector<ALCCandidate> candidates, const GraphState& graph,
    const SaliencyState& saliency_state,
    const nav_msgs::msg::OccupancyGrid& map) const {
    if (candidates.empty() || graph.robot_ix < 0) {
        return std::nullopt;
    }

    if (graph.robot_ix >= static_cast<int>(graph.keyframes.size())) {
        return std::nullopt;
    }

    const Eigen::Vector3f& robot_pos =
        graph.keyframes[static_cast<std::size_t>(graph.robot_ix)].pose.position;
    buildSaliencyOverlay(graph, saliency_state, map, saliency_overlay_scratch_);
    std::sort(candidates.begin(), candidates.end(),
              [](const ALCCandidate& lhs, const ALCCandidate& rhs) {
                  return lhs.reward_ub > rhs.reward_ub;
              });

    std::optional<ALCCandidate> best_candidate;
    for (auto& candidate : candidates) {
        if (best_candidate.has_value() &&
            candidate.reward_ub <= best_candidate->reward) {
            break;
        }

        candidate.map_dist = path_planner_.computeDist(
            robot_pos, candidate.rep_pose.position, map,
            saliency_overlay_scratch_.empty() ? nullptr
                                              : &saliency_overlay_scratch_);
        if (!std::isfinite(candidate.map_dist)) {
            continue;
        }

        evaluator_.fillReward(candidate, saliency_state);
        if (!best_candidate.has_value() ||
            candidate.reward > best_candidate->reward) {
            best_candidate = candidate;
        }
    }

    return best_candidate;
}

}  // namespace alc_planner
