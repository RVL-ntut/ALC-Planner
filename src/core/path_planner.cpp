#include "alc_planner/path_planner.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <vector>

namespace alc_planner
{

namespace
{

struct Cell
{
    int row = 0;
    int col = 0;
};

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

bool inBounds(const Cell& cell, const int width, const int height) {
    return cell.col >= 0 && cell.col < width && cell.row >= 0 &&
           cell.row < height;
}

int toIndex(const Cell& cell, const int width) {
    return cell.row * width + cell.col;
}

Cell toCell(const Eigen::Vector3f& pos,
            const nav_msgs::msg::OccupancyGrid& map) {
    const float resolution = map.info.resolution;
    const float origin_x = static_cast<float>(map.info.origin.position.x);
    const float origin_y = static_cast<float>(map.info.origin.position.y);
    return {
        static_cast<int>(std::floor((pos.y() - origin_y) / resolution)),
        static_cast<int>(std::floor((pos.x() - origin_x) / resolution)),
    };
}

float heuristic(const Cell& from, const Cell& to, const float resolution) {
    const float dx = static_cast<float>(from.col - to.col);
    const float dy = static_cast<float>(from.row - to.row);
    return std::sqrt(dx * dx + dy * dy) * resolution;
}

}  // namespace

float PathPlanner::computeDist(const Eigen::Vector3f& start,
                               const Eigen::Vector3f& goal,
                               const nav_msgs::msg::OccupancyGrid& map) const {
    const int width = static_cast<int>(map.info.width);
    const int height = static_cast<int>(map.info.height);
    if (width <= 0 || height <= 0 || map.info.resolution <= 0.0f) {
        return std::numeric_limits<float>::infinity();
    }

    const Cell start_cell = toCell(start, map);
    const Cell goal_cell = toCell(goal, map);
    if (!inBounds(start_cell, width, height) ||
        !inBounds(goal_cell, width, height)) {
        return std::numeric_limits<float>::infinity();
    }

    if (start_cell.row == goal_cell.row && start_cell.col == goal_cell.col) {
        return 0.0f;
    }

    const auto isFree = [&](const Cell& cell) {
        const int8_t value =
            map.data[static_cast<std::size_t>(toIndex(cell, width))];
        return value >= 0 && value <= kOccupiedThreshold;
    };
    if (!isFree(start_cell) || !isFree(goal_cell)) {
        return std::numeric_limits<float>::infinity();
    }

    const int cell_count = width * height;
    if (static_cast<std::size_t>(cell_count) > map.data.size()) {
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

        const Cell current_cell{current.index / width, current.index % width};
        for (const auto& offset : kNeighborOffsets) {
            const Cell neighbor{current_cell.row + offset[0],
                                current_cell.col + offset[1]};
            if (!inBounds(neighbor, width, height) || !isFree(neighbor)) {
                continue;
            }

            const float step_cost = (offset[0] != 0 && offset[1] != 0)
                                        ? std::sqrt(2.0f) * map.info.resolution
                                        : map.info.resolution;
            const int neighbor_index = toIndex(neighbor, width);
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
