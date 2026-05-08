#include "alc_planner/path_planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <vector>

#include "alc_planner/map_utils.hpp"

namespace alc_planner
{

namespace
{

struct QueueEntry
{
    float f_cost = 0.0f;
    float g_cost = 0.0f;
    int index = 0;
};

struct QueueCompare
{
    bool operator()(const QueueEntry& lhs, const QueueEntry& rhs) const {
        return lhs.f_cost > rhs.f_cost;
    }
};

float heuristic(const GridCell& from, const GridCell& to,
                const float resolution) {
    const float dx = static_cast<float>(from.col - to.col);
    const float dy = static_cast<float>(from.row - to.row);
    return std::sqrt(dx * dx + dy * dy) * resolution;
}

}  // namespace

float PathPlanner::computeDist(
    const Eigen::Vector3f& start, const Eigen::Vector3f& goal,
    const nav_msgs::msg::OccupancyGrid& map,
    const std::vector<float>* saliency_overlay) const {
    const int width = static_cast<int>(map.info.width);
    const int height = static_cast<int>(map.info.height);
    if (width <= 0 || height <= 0 || map.info.resolution <= 0.0f) {
        return std::numeric_limits<float>::infinity();
    }

    const GridCell start_cell = toCell(start, map);
    const GridCell goal_cell = toCell(goal, map);
    if (!inBounds(start_cell, width, height) ||
        !inBounds(goal_cell, width, height)) {
        return std::numeric_limits<float>::infinity();
    }

    if (start_cell.row == goal_cell.row && start_cell.col == goal_cell.col) {
        return 0.0f;
    }

    const int cell_count = width * height;
    if (static_cast<std::size_t>(cell_count) > map.data.size()) {
        return std::numeric_limits<float>::infinity();
    }

    const auto isFree = [&](const GridCell& cell) {
        const int8_t value =
            map.data[static_cast<std::size_t>(toIndex(cell, width))];
        return value >= 0 && value <= kOccupiedThreshold;
    };
    if (!isFree(start_cell) || !isFree(goal_cell)) {
        return std::numeric_limits<float>::infinity();
    }

    std::vector<float> g_costs(static_cast<std::size_t>(cell_count),
                               std::numeric_limits<float>::infinity());
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueCompare>
        open_set;

    const int start_index = toIndex(start_cell, width);
    const int goal_index = toIndex(goal_cell, width);
    g_costs[static_cast<std::size_t>(start_index)] = 0.0f;
    open_set.push({heuristic(start_cell, goal_cell, map.info.resolution), 0.0f,
                   start_index});

    constexpr int kNeighborOffsets[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1},
    };

    while (!open_set.empty()) {
        const QueueEntry current = open_set.top();
        open_set.pop();

        if (current.g_cost > g_costs[static_cast<std::size_t>(current.index)]) {
            continue;
        }
        if (current.index == goal_index) {
            return current.g_cost;
        }

        const GridCell current_cell{current.index / width,
                                    current.index % width};
        for (const auto& offset : kNeighborOffsets) {
            const GridCell neighbor{current_cell.row + offset[0],
                                    current_cell.col + offset[1]};
            if (!inBounds(neighbor, width, height) || !isFree(neighbor)) {
                continue;
            }

            const int neighbor_index = toIndex(neighbor, width);
            float weight = 1.0f;
            if (saliency_overlay != nullptr &&
                static_cast<std::size_t>(neighbor_index) <
                    saliency_overlay->size()) {
                const float saliency =
                    (*saliency_overlay)[static_cast<std::size_t>(
                        neighbor_index)];
                weight = 2.0f - std::clamp(saliency, 0.0f, 1.0f);
            }
            const float step_cost =
                weight * ((offset[0] != 0 && offset[1] != 0)
                              ? std::sqrt(2.0f) * map.info.resolution
                              : map.info.resolution);
            const float candidate_cost = current.g_cost + step_cost;
            float& known_cost =
                g_costs[static_cast<std::size_t>(neighbor_index)];
            if (candidate_cost >= known_cost) {
                continue;
            }

            known_cost = candidate_cost;
            open_set.push({candidate_cost + heuristic(neighbor, goal_cell,
                                                      map.info.resolution),
                           candidate_cost, neighbor_index});
        }
    }

    return std::numeric_limits<float>::infinity();
}

}  // namespace alc_planner
