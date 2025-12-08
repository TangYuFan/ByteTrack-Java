# ByteTrack Java

这是 ByteTrack 的 Java 实现，核心模块保持原版逻辑，包括三阶段 Track 管理、8D Kalman Filter、匈牙利匹配等。

## Demo

<p align="center">
  <img src="demo.png" alt="ByteTrack Demo">
</p>


## 算法实现对比

| 模块 / 功能           | 原版 ByteTrack                                               | 当前 Java 实现                    | 差异 / 影响          |
| ----------------- | ---------------------------------------------------------- | ------------------------------ | ---------------- |
| **Track 管理**      | 三阶段：Tentative → Active → Lost；hitStreak 达到阈值激活，lost 超过阈值删除 | 完全相同三阶段逻辑                      | 核心逻辑一致           |
| **Kalman Filter** | 8D `[x, y, a, h, vx, vy, va, vh]`，F/H/P/Q/R 矩阵             | 同样 8D KF，F/H/P/Q/R 矩阵一致        | 一致               |
| **IOU 匹配**        | 用矩形 IOU 计算匹配代价                                             | 同样使用矩形 IOU                     | 一致               |
| **匈牙利匹配**         | 使用成熟库/linear_sum_assignment                                | 自实现匈牙利算法                       | 算法结果一致，但性能可能略低   |
| **高置信度匹配**        | 匹配后更新 KF，hitStreak++，状态激活                                  | 相同逻辑                           | 一致               |
| **低置信度匹配**        | Tentative Track 更新，classId + gating                        | 更新策略略不同（只匹配 classId + low IOU） | 对追踪精度影响较小，但策略略简化 |




## 官方原版实现 JNI WAPPER

// 对应的 JAVA 接口
package com.example.bytetrack;
import java.util.List;
public class ByteTrackJni {
    static {
        System.loadLibrary("bytetrack"); // libbytetrack.so
    }
public native void initTracker(int fps, int trackBuffer);
public native List<float[]> updateTracker(float[] detections, float[] probs);
public native void releaseTracker();
}


// 对应的 JAVA 调用示例
ByteTrackJni tracker = new ByteTrackJni();       // 跟踪器，注意非多线程安全
tracker.initTracker(30, 30);                     // fps：视频帧率（用于卡尔曼滤波预测），trackBuffer：目标丢失多少帧后删除轨迹
float[] dets = {x1, y1, w1, h1, x2, y2, w2, h2}; // 每帧检测结果目标边框， xywh、xywh、xywh ..（注意这里输入非归一化后的位置坐标信息）
float[] probs = {0.9f, 0.8f};                    // 每帧检测结果目标置信度：score、score ..
List<float[]> tracks = tracker.updateTracker(dets, probs);   // 输出跟踪后的id、置信度和边框
for (float[] t : tracks) {
    int trackId = (int) t[0];     // id
    float prob = t[1];            // 置信度
    float x = t[2], y = t[3], w = t[4], h = t[5]; // 边框
}
tracker.releaseTracker();                         // 销毁跟踪器