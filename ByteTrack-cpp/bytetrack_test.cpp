#include "ByteTrack/BYTETracker.h"
#include "gtest/gtest.h"

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/foreach.hpp"
#include "boost/optional.hpp"

#include <cstddef> // for size_t

namespace
{
    // 误差容忍度（因为浮点数计算可能有微小误差）
    constexpr double EPS = 1e-2;

    // 两个输入文件（检测结果和跟踪结果）
    const std::string D_RESULTS_FILE = "./data/detection_results.json";  // 目标检测结果。作为 ByteTrack 的输入
    const std::string T_RESULTS_FILE = "./data/tracking_results.json";   // 官方 ByteTrack 的试过一次关于，用来验证测测试结果

    // typedef：BYTETracker 输出格式
    // key: track_id (跟踪目标ID), value: Rect(目标矩形框)
    using BYTETrackerOut = std::map<size_t, byte_track::Rect<float>>;

    // -----------------------------
    // 通用函数：从 property_tree 中读取某个 key 的数据
    // 如果 key 不存在，则抛出异常
    // -----------------------------
    template <typename T>
    T get_data(const boost::property_tree::ptree &pt, const std::string &key)
    {
        T ret;
        if (boost::optional<T> data = pt.get_optional<T>(key))
        {
            ret = data.get();
        }
        else
        {
            throw std::runtime_error("Could not read the data from ptree: [key: " + key + "]");
        }
        return ret;
    }

    // -----------------------------
    // 从检测结果 JSON（detection_results.json）中读取输入数据
    // 生成 map<frame_id, vector<Object>>
    // 即每一帧对应多个检测到的目标对象
    // -----------------------------
    std::map<size_t, std::vector<byte_track::Object>> get_inputs_ref(const boost::property_tree::ptree &pt)
    {
        std::map<size_t, std::vector<byte_track::Object>> inputs_ref;

        // 遍历 "results" 数组节点
        BOOST_FOREACH (const boost::property_tree::ptree::value_type &child, pt.get_child("results"))
        {
            const boost::property_tree::ptree &result = child.second;
            // 读取 JSON 中的各个字段
            const auto frame_id = get_data<int>(result, "frame_id");
            const auto prob = get_data<float>(result, "prob");
            const auto x = get_data<float>(result, "x");
            const auto y = get_data<float>(result, "y");
            const auto width = get_data<float>(result, "width");
            const auto height = get_data<float>(result, "height");

            // 将 Object 添加到对应帧的列表中
            decltype(inputs_ref)::iterator itr = inputs_ref.find(frame_id);
            if (itr != inputs_ref.end())
            {
                itr->second.emplace_back(byte_track::Rect(x, y, width, height), 0, prob);
            }
            else
            {
                std::vector<byte_track::Object> v(1, {byte_track::Rect(x, y, width, height), 0, prob});
                inputs_ref.emplace_hint(inputs_ref.end(), frame_id, v);
            }
        }
        return inputs_ref;
    }

    // -----------------------------
    // 从跟踪结果 JSON（tracking_results.json）中读取参考输出
    // 生成 map<frame_id, map<track_id, Rect>>
    // 即每一帧中，每个 track_id 对应一个矩形框位置
    // -----------------------------
    std::map<size_t, BYTETrackerOut> get_outputs_ref(const boost::property_tree::ptree &pt)
    {
        std::map<size_t, BYTETrackerOut> outputs_ref;

        BOOST_FOREACH (const boost::property_tree::ptree::value_type &child, pt.get_child("results"))
        {
            const boost::property_tree::ptree &result = child.second;
            const auto frame_id = get_data<int>(result, "frame_id");
            const auto track_id = get_data<int>(result, "track_id");
            const auto x = get_data<float>(result, "x");
            const auto y = get_data<float>(result, "y");
            const auto width = get_data<float>(result, "width");
            const auto height = get_data<float>(result, "height");

            decltype(outputs_ref)::iterator itr = outputs_ref.find(frame_id);
            if (itr != outputs_ref.end())
            {
                itr->second.emplace(track_id, byte_track::Rect<float>(x, y, width, height));
            }
            else
            {
                BYTETrackerOut v{
                    {track_id, byte_track::Rect<float>(x, y, width, height)},
                };
                outputs_ref.emplace_hint(outputs_ref.end(), frame_id, v);
            }
        }
        return outputs_ref;
    }
} // namespace

// -----------------------------
// 主测试用例：测试 BYTETracker 是否与官方结果一致
// -----------------------------
TEST(ByteTrack, BYTETracker)
{
    boost::property_tree::ptree pt_d_results;
    boost::property_tree::read_json(D_RESULTS_FILE, pt_d_results);

    boost::property_tree::ptree pt_t_results;
    boost::property_tree::read_json(T_RESULTS_FILE, pt_t_results);

    try
    {
        // 从 JSON 文件中读取元信息
        const auto detection_results_name = get_data<std::string>(pt_d_results, "name");
        const auto tracking_results_name = get_data<std::string>(pt_t_results, "name");
        const auto fps = get_data<int>(pt_d_results, "fps");
        const auto track_buffer = get_data<int>(pt_d_results, "track_buffer");

        // 确保检测结果和跟踪结果是同一组数据
        if (detection_results_name != tracking_results_name)
        {
            throw std::runtime_error("The name of the tests are different: [detection_results_name: " + detection_results_name +
                                     ", tracking_results_name: " + tracking_results_name + "]");
        }

        // 读取检测输入（输入帧数据）
        const auto inputs_ref = get_inputs_ref(pt_d_results);

        // 读取官方参考输出（正确的跟踪结果）
        auto outputs_ref = get_outputs_ref(pt_t_results);

        // 初始化 BYTETracker
        byte_track::BYTETracker tracker(fps, track_buffer);

        // 遍历每一帧进行测试
        for (const auto &[frame_id, objects] : inputs_ref)
        {
            // 执行跟踪更新
            const auto outputs = tracker.update(objects);

            // 验证：当前帧输出的目标数量与参考数据相同
            EXPECT_EQ(outputs.size(), outputs_ref[frame_id].size());

            // 对每个目标，逐项比较坐标值
            for (const auto &outputs_per_frame : outputs)
            {
                const auto &rect = outputs_per_frame->getRect();
                const auto &track_id = outputs_per_frame->getTrackId();
                const auto &ref = outputs_ref[frame_id][track_id];
                EXPECT_NEAR(ref.x(), rect.x(), EPS);
                EXPECT_NEAR(ref.y(), rect.y(), EPS);
                EXPECT_NEAR(ref.width(), rect.width(), EPS);
                EXPECT_NEAR(ref.height(), rect.height(), EPS);
            }
        }
    }
    catch (const std::exception &e)
    {
        FAIL() << e.what(); // 如果过程中有异常，测试失败并打印错误
    }
}

// -----------------------------
// 主函数入口：运行所有 GTest 测试
// 使用流程：
//      步骤 1：初始化跟踪器：byte_track::BYTETracker tracker(fps, track_buffer);
//      步骤 2：逐帧传入检测结果：
//          for (const auto &[frame_id, objects] : inputs_ref)
//          {
//              const auto outputs = tracker.update(objects);
//          }
//      步骤 3：获取跟踪结果
//         for (const auto &outputs_per_frame : outputs)
//          {
//               const auto &rect = outputs_per_frame->getRect();
//             const auto &track_id = outputs_per_frame->getTrackId();
//          }
// -----------------------------
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return (RUN_ALL_TESTS());
}
