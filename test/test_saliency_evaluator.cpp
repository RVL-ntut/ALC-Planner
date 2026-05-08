#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

#include "alc_planner/saliency_evaluator.hpp"
#include "alc_planner/types.hpp"

namespace alc_planner
{
namespace
{

Keyframe makeKeyframe(const int node_id, std::vector<int32_t> word_ids) {
    Keyframe keyframe;
    keyframe.node_id = node_id;
    keyframe.word_ids = std::move(word_ids);
    return keyframe;
}

}  // namespace

TEST(SaliencyEvaluator, SLNormalizesCorrectly) {
    Params params;
    GraphState graph;
    std::vector<int32_t> words_100(100);
    std::vector<int32_t> words_50(50);
    std::iota(words_100.begin(), words_100.end(), 1);
    std::iota(words_50.begin(), words_50.end(), 101);

    graph.keyframes.emplace(1, makeKeyframe(1, std::move(words_100)));
    graph.keyframes.emplace(2, makeKeyframe(2, std::move(words_50)));

    SaliencyEvaluator evaluator(params);
    evaluator.observeWordsRecognized(100);
    evaluator.update(graph);

    EXPECT_NEAR(graph.keyframes.at(1).saliency_local, 1.0f, 1e-4f);
    EXPECT_NEAR(graph.keyframes.at(2).saliency_local, 0.5f, 1e-4f);
    EXPECT_GT(graph.keyframes.at(1).plc_intrinsic,
              graph.keyframes.at(2).plc_intrinsic);
}

TEST(SaliencyEvaluator, SGRarityOrder) {
    Params params;
    GraphState graph;
    graph.keyframes.emplace(1, makeKeyframe(1, {1, 2}));
    graph.keyframes.emplace(2, makeKeyframe(2, {3, 4}));
    graph.keyframes.emplace(3, makeKeyframe(3, {3, 4, 5, 6}));

    SaliencyEvaluator evaluator(params);
    evaluator.observeWordsRecognized(4);
    evaluator.update(graph);

    EXPECT_GT(graph.keyframes.at(1).saliency_global,
              graph.keyframes.at(2).saliency_global);
}

TEST(SaliencyEvaluator, EmptyWordIdsNoCrash) {
    Params params;
    GraphState graph;
    graph.keyframes.emplace(1, makeKeyframe(1, {}));
    graph.robot_node_id = 1;

    SaliencyEvaluator evaluator(params);
    evaluator.update(graph);

    EXPECT_FLOAT_EQ(graph.keyframes.at(1).saliency_local, 0.0f);
    EXPECT_FLOAT_EQ(graph.keyframes.at(1).saliency_global, 0.0f);
    EXPECT_FLOAT_EQ(graph.keyframes.at(1).plc_intrinsic, 0.0f);
    EXPECT_FLOAT_EQ(evaluator.latestSL(), 0.0f);
}

TEST(SaliencyEvaluator, SingleNodeSGIsZero) {
    Params params;
    GraphState graph;
    graph.keyframes.emplace(1, makeKeyframe(1, {10, 20, 30}));

    SaliencyEvaluator evaluator(params);
    evaluator.update(graph);

    EXPECT_FLOAT_EQ(graph.keyframes.at(1).saliency_global, 0.0f);
}

}  // namespace alc_planner
