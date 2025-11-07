# ByteTrack Java

这是 ByteTrack 的 Java 实现，核心模块保持原版逻辑，包括三阶段 Track 管理、8D Kalman Filter、匈牙利匹配等。

## 算法实现对比

| 模块 / 功能           | 原版 ByteTrack                                               | 当前 Java 实现                    | 差异 / 影响          |
| ----------------- | ---------------------------------------------------------- | ------------------------------ | ---------------- |
| **Track 管理**      | 三阶段：Tentative → Active → Lost；hitStreak 达到阈值激活，lost 超过阈值删除 | 完全相同三阶段逻辑                      | 核心逻辑一致           |
| **Kalman Filter** | 8D `[x, y, a, h, vx, vy, va, vh]`，F/H/P/Q/R 矩阵             | 同样 8D KF，F/H/P/Q/R 矩阵一致        | 一致               |
| **IOU 匹配**        | 用矩形 IOU 计算匹配代价                                             | 同样使用矩形 IOU                     | 一致               |
| **匈牙利匹配**         | 使用成熟库/linear_sum_assignment                                | 自实现匈牙利算法                       | 算法结果一致，但性能可能略低   |
| **高置信度匹配**        | 匹配后更新 KF，hitStreak++，状态激活                                  | 相同逻辑                           | 一致               |
| **低置信度匹配**        | Tentative Track 更新，classId + gating                        | 更新策略略不同（只匹配 classId + low IOU） | 对追踪精度影响较小，但策略略简化 |


## Demo

下面是 ByteTrack Java 运行效果示例：

<p align="center">
  <img src="demo.gif" alt="ByteTrack Demo">
</p>

