#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "alc_planner/types.hpp"
#include "alc_planner/uncertainty_metrics.hpp"

namespace alc_planner
{
namespace
{

void addNode(GraphState& graph, const int node_id) {
    Keyframe keyframe;
    keyframe.node_id = node_id;
    graph.keyframes.emplace(node_id, keyframe);
    graph.adj[node_id];
}

void addDirectedEdge(GraphState& graph, const int from_id, const int to_id,
                     const float dist) {
    graph.adj[from_id].push_back({to_id, dist});
}

}  // namespace

TEST(UncertaintyMetrics, DijkstraKnownGraph) {
    GraphState graph;
    for (int node_id = 0; node_id <= 4; ++node_id) {
        addNode(graph, node_id);
    }

    addDirectedEdge(graph, 0, 1, 1.0f);
    addDirectedEdge(graph, 1, 2, 2.0f);
    addDirectedEdge(graph, 2, 3, 0.5f);
    addDirectedEdge(graph, 1, 4, 1.0f);
    addDirectedEdge(graph, 4, 3, 3.0f);

    const auto dist_map = UncertaintyMetrics::dijkstraAll(graph, 0);
    ASSERT_EQ(dist_map.size(), 5U);
    EXPECT_NEAR(dist_map.at(1), 1.0f, 1e-4f);
    EXPECT_NEAR(dist_map.at(2), 3.0f, 1e-4f);
    EXPECT_NEAR(dist_map.at(4), 2.0f, 1e-4f);
    EXPECT_NEAR(UncertaintyMetrics::graphDist(graph, 0, 3), 3.5f, 1e-4f);
}

TEST(UncertaintyMetrics, UnreachableReturnsInf) {
    GraphState graph;
    addNode(graph, 0);
    addNode(graph, 1);
    addNode(graph, 2);
    addDirectedEdge(graph, 0, 1, 1.0f);

    EXPECT_TRUE(std::isinf(UncertaintyMetrics::graphDist(graph, 0, 2)));
}

TEST(UncertaintyMetrics, LoopClosureEdgeShortcutsPath) {
    GraphState graph;
    for (int node_id = 0; node_id <= 3; ++node_id) {
        addNode(graph, node_id);
    }

    addDirectedEdge(graph, 0, 1, 2.0f);
    addDirectedEdge(graph, 1, 2, 2.0f);
    addDirectedEdge(graph, 2, 3, 2.0f);
    addDirectedEdge(graph, 0, 3, 1.2f);

    EXPECT_NEAR(UncertaintyMetrics::graphDist(graph, 0, 3), 1.2f, 1e-4f);
}

TEST(UncertaintyMetrics, SourceNotInGraphReturnsInf) {
    GraphState graph;
    addNode(graph, 1);
    addNode(graph, 2);
    addDirectedEdge(graph, 1, 2, 1.0f);

    EXPECT_TRUE(UncertaintyMetrics::dijkstraAll(graph, 99).empty());
    EXPECT_TRUE(std::isinf(UncertaintyMetrics::graphDist(graph, 99, 1)));
}

}  // namespace alc_planner
