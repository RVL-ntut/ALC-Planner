#pragma once

#include <Eigen/Core>
#include <nav_msgs/msg/occupancy_grid.hpp>

namespace alc_planner
{

class PathPlanner
{
public:
    float computeDist(const Eigen::Vector3f& start, const Eigen::Vector3f& goal,
                      const nav_msgs::msg::OccupancyGrid& map) const;

private:
    static constexpr int kOccupiedThreshold = 50;
};

}  // namespace alc_planner
