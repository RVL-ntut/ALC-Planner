#include <Eigen/Core>
#include <gtest/gtest.h>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include <cmath>
#include <limits>
#include <vector>

#include "alc_planner/path_planner.hpp"

namespace alc_planner
{
namespace
{

nav_msgs::msg::OccupancyGrid buildGrid(const int width, const int height,
                                       const float resolution) {
    nav_msgs::msg::OccupancyGrid map;
    map.info.width = width;
    map.info.height = height;
    map.info.resolution = resolution;
    map.info.origin.orientation.w = 1.0;
    map.data.assign(static_cast<std::size_t>(width * height), 0);
    return map;
}

void setCell(nav_msgs::msg::OccupancyGrid& map, const int row, const int col,
             const int8_t value) {
    const int width = static_cast<int>(map.info.width);
    map.data[static_cast<std::size_t>(row * width + col)] = value;
}

Eigen::Vector3f pos(const float x, const float y) {
    return Eigen::Vector3f(x, y, 0.0f);
}

}  // namespace

TEST(PathPlanner, OpenFieldHorizontal) {
    const auto map = buildGrid(20, 20, 1.0f);
    PathPlanner planner;

    const float dist =
        planner.computeDist(pos(0.5f, 0.5f), pos(5.5f, 0.5f), map);
    EXPECT_NEAR(dist, 5.0f, 1e-4f);
}

TEST(PathPlanner, SameCell) {
    const auto map = buildGrid(20, 20, 1.0f);
    PathPlanner planner;

    const float dist =
        planner.computeDist(pos(1.2f, 1.2f), pos(1.8f, 1.4f), map);
    EXPECT_FLOAT_EQ(dist, 0.0f);
}

TEST(PathPlanner, DiagonalPath) {
    const auto map = buildGrid(20, 20, 1.0f);
    PathPlanner planner;

    const float dist =
        planner.computeDist(pos(0.5f, 0.5f), pos(3.5f, 3.5f), map);
    EXPECT_NEAR(dist, 3.0f * std::sqrt(2.0f), 1e-4f);
}

TEST(PathPlanner, WallObstacle) {
    auto map = buildGrid(7, 7, 1.0f);
    for (int row = 0; row < 7; ++row) {
        if (row != 5) {
            setCell(map, row, 3, 100);
        }
    }

    PathPlanner planner;
    const float dist =
        planner.computeDist(pos(0.5f, 3.5f), pos(6.5f, 3.5f), map);
    EXPECT_TRUE(std::isfinite(dist));
    EXPECT_GT(dist, 6.0f);
}

TEST(PathPlanner, CompletelyBlocked) {
    auto map = buildGrid(5, 5, 1.0f);
    setCell(map, 1, 2, 100);
    setCell(map, 2, 1, 100);
    setCell(map, 2, 3, 100);
    setCell(map, 3, 2, 100);
    setCell(map, 1, 1, 100);
    setCell(map, 1, 3, 100);
    setCell(map, 3, 1, 100);
    setCell(map, 3, 3, 100);

    PathPlanner planner;
    const float dist =
        planner.computeDist(pos(0.5f, 2.5f), pos(2.5f, 2.5f), map);
    EXPECT_TRUE(std::isinf(dist));
}

TEST(PathPlanner, OutOfBoundsGoal) {
    const auto map = buildGrid(5, 5, 1.0f);
    PathPlanner planner;

    const float dist =
        planner.computeDist(pos(0.5f, 0.5f), pos(10.0f, 0.5f), map);
    EXPECT_TRUE(std::isinf(dist));
}

TEST(PathPlanner, UnknownCells) {
    auto map = buildGrid(5, 5, 1.0f);
    setCell(map, 2, 2, -1);

    PathPlanner planner;
    const float dist =
        planner.computeDist(pos(0.5f, 2.5f), pos(4.5f, 2.5f), map);
    EXPECT_TRUE(std::isfinite(dist));
    EXPECT_GT(dist, 4.0f);
}

TEST(PathPlanner, ZeroSaliencyOverlayDoublesCost) {
    const auto map = buildGrid(20, 20, 1.0f);
    std::vector<float> overlay(map.data.size(), 0.0f);
    PathPlanner planner;

    const float dist =
        planner.computeDist(pos(0.5f, 0.5f), pos(5.5f, 0.5f), map, &overlay);
    EXPECT_NEAR(dist, 10.0f, 1e-4f);
}

TEST(PathPlanner, UnitSaliencyOverlayMatchesStandardAStar) {
    const auto map = buildGrid(20, 20, 1.0f);
    std::vector<float> overlay(map.data.size(), 1.0f);
    PathPlanner planner;

    const float standard_dist =
        planner.computeDist(pos(0.5f, 0.5f), pos(5.5f, 0.5f), map);
    const float weighted_dist =
        planner.computeDist(pos(0.5f, 0.5f), pos(5.5f, 0.5f), map, &overlay);
    EXPECT_NEAR(weighted_dist, standard_dist, 1e-4f);
}

TEST(PathPlanner, SaliencyWeightedAStarPrefersLowerTotalWeightedCost) {
    const auto map = buildGrid(5, 3, 1.0f);
    std::vector<float> overlay(map.data.size(), 0.0f);
    const int width = static_cast<int>(map.info.width);

    for (int col = 0; col < width; ++col) {
        overlay[static_cast<std::size_t>(0 * width + col)] = 1.0f;
    }
    overlay[static_cast<std::size_t>(1 * width + 0)] = 1.0f;
    overlay[static_cast<std::size_t>(1 * width + 4)] = 1.0f;

    PathPlanner planner;
    const float dist =
        planner.computeDist(pos(0.5f, 1.5f), pos(4.5f, 1.5f), map, &overlay);
    EXPECT_GT(dist, 4.0f);
    EXPECT_LT(dist, 8.0f);
}

}  // namespace alc_planner
