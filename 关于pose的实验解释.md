你这里的“相机位姿估计实验”，核心指的是 **VGGT 的 `camera_head` 直接根据多张图像预测相机外参，然后用真值位姿评估预测结果**。它和“先匹配特征点，再用 `recoverPose` 算位姿”的实验不是一回事。

主要对应：

- 实验总结：[euroc_v103_three_groups_experiment.md](/e:/vgg_xzy/euroc_v103_three_groups_experiment.md)
- EuRoC 实验脚本：[eval_euroc_v103.py](/e:/vgg_xzy/vggt/tools/eval_euroc_v103.py:493)
- 复用的 CO3D 位姿评测代码：[eval_co3d_hydrant_camera_pose.py](/e:/vgg_xzy/vggt/tools/eval_co3d_hydrant_camera_pose.py:100)
- VGGT 相机头：[camera_head.py](/e:/vgg_xzy/vggt/vggt/heads/camera_head.py:19)

**一、实验到底在测什么**

实验想回答的问题是：

> 给 VGGT 一段连续图像，它能不能直接推断出这些图像之间的相机运动关系？

这里并不要求恢复真实世界中的绝对坐标，而是看任意两帧之间的：

- 相对旋转是否正确；
- 相对平移方向是否正确。

原因是单目图像天然存在尺度不确定性：模型可能知道相机往哪个方向移动了，但无法确定真实移动了多少米。因此实验评估的是 **旋转角误差** 和 **平移方向角误差**，而不是平移距离误差。

---

**二、实验数据怎么准备**

当前报告重点使用的是 **EuRoC MAV 数据集的 V103 序列，`cam0` 相机**。

EuRoC 提供：

- 相机图像；
- 相机内参与畸变参数；
- IMU/body 的高精度轨迹真值；
- 相机相对于 body 的外参。

脚本首先读取：

```text
mav0/cam0/data.csv
mav0/cam0/sensor.yaml
mav0/state_groundtruth_estimate0/data.csv
```

具体逻辑在 [eval_euroc_v103.py](/e:/vgg_xzy/vggt/tools/eval_euroc_v103.py:169)。

每张相机图像有一个时间戳，脚本为它寻找时间上最近的真值轨迹，且要求二者时间差不超过默认的 `3 ms`：

```python
--max_gt_dt_ns 3000000
```

这样每张图像都能得到对应的真值相机位姿。

---

**三、真值相机位姿是怎样得到的**

EuRoC 给出的真值主要是 body 在世界坐标系中的姿态，而实验需要的是相机的外参。

脚本先构造 body-to-world 变换：

```text
T_WB
```

再从相机标定文件读取相机到 body 的固定变换：

```text
T_BC
```

于是相机到世界的变换为：

```text
T_WC = T_WB · T_BC
```

最后取逆，得到评测使用的 OpenCV 风格外参：

```text
T_CW = inverse(T_WC)
```

也就是：

```text
世界坐标 -> 相机坐标
```

形式为：

```text
[R | t]
```

这部分实现在 [eval_euroc_v103.py](/e:/vgg_xzy/vggt/tools/eval_euroc_v103.py:194)。

---

**四、为什么要选三组不同运动强度的片段**

实验没有随便截一段视频，而是刻意选择了三组运动条件：

| 片段 | 特点 | path length | baseline | 累计旋转 |
|---|---|---:|---:|---:|
| 开头静态 | 几乎不移动 | 0.0035 m | 0.0016 m | 0.54 deg |
| 中段大运动 A | 平移与旋转都明显 | 4.6103 m | 2.4082 m | 183.73 deg |
| 中段大运动 B | 大幅旋转，存在有效运动 | 4.5398 m | 0.4104 m | 216.72 deg |

每组取 `10` 帧，`frame_stride=10`。因为 EuRoC 图像流约为 `20 Hz`，所以相邻选中帧间隔约 `0.5 s`，整组跨度约 `4.5 s`。

选帧时，脚本计算每个窗口的运动分数：

```text
score = path_length + 0.5 × baseline + 0.1 × rotation / 180
```

然后：

- 静态组选择运动最小的窗口；
- 运动组选择运动最大的窗口。

代码见 [eval_euroc_v103.py](/e:/vgg_xzy/vggt/tools/eval_euroc_v103.py:217)。

这样做非常重要，因为位姿估计对运动条件敏感：

- 几乎静止时，平移方向不可可靠估计；
- 有足够视角变化时，才能真正考察模型能否恢复相机运动。

---

**五、图像输入前做了什么处理**

EuRoC 原始图像存在镜头畸变。脚本先根据相机标定参数对选中图像去畸变：

```python
cv2.undistort(...)
```

并同时生成去畸变后的新内参矩阵 `K`。

代码见 [eval_euroc_v103.py](/e:/vgg_xzy/vggt/tools/eval_euroc_v103.py:283)。

虽然直接 Camera Pose 指标主要用外参评估，但去畸变仍然有意义：

- 输入图像的几何关系更接近针孔相机模型；
- 与同脚本中的 matching / epipolar 评测保持一致；
- 避免畸变对视觉运动判断造成额外影响。

---

**六、VGGT 如何直接预测相机位姿**

这一部分是实验的核心。

对于一组 `10` 张图像，脚本将它们一次性送入 VGGT：

```python
aggregated_tokens_list, _ = self.model.aggregator(images_batched)
pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
pred_extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
```

代码见 [eval_euroc_v103.py](/e:/vgg_xzy/vggt/tools/eval_euroc_v103.py:493)。

流程可以理解成：

```text
10 张图像
  -> VGGT aggregator 提取跨视图特征
  -> camera tokens 表示每一帧的相机信息
  -> camera_head 迭代预测相机参数
  -> 输出每帧预测外参 [R | t]
```

`camera_head` 输出的是一个 9 维相机编码：

```text
[T_x, T_y, T_z, q_w, q_x, q_y, q_z, FoV_h, FoV_w]
```

其中：

- 前 3 维：平移；
- 中间 4 维：四元数形式的旋转；
- 后 2 维：水平和垂直视场角。

定义见 [pose_enc.py](/e:/vgg_xzy/vggt/vggt/utils/pose_enc.py:62)。

相机头不是一次直接猜结果，而是进行默认 `4` 次迭代修正：

```text
初始空位姿
 -> 第一次预测
 -> 基于上一次预测继续修正
 -> ...
 -> 第四次输出最终位姿
```

实现见 [camera_head.py](/e:/vgg_xzy/vggt/vggt/heads/camera_head.py:95)。

需要特别注意：

> Camera Pose 实验没有使用 ALIKED、没有匹配点、没有 Essential Matrix、没有 RANSAC、也没有 `recoverPose`。它评测的是 VGGT `camera_head` 的直接位姿预测能力。

---

**七、为什么不直接比较每一帧的绝对外参**

模型预测出的整组相机可能位于一个与真值不同的全局坐标系中。例如，整组预测整体旋转了一下，或者整体平移了一下，但帧与帧之间的运动关系仍然正确。

因此实验不直接比较：

```text
预测第 i 帧外参 vs 真值第 i 帧外参
```

而是比较任意两帧之间的相对位姿：

```text
预测相对位姿 vs 真值相对位姿
```

对于 `10` 帧，一共有：

```text
C(10, 2) = 45 对
```

所以 Camera Pose 表格里的结果来自每组 `45` 个 frame pair，而不是只评测相邻的 `9` 对。

计算代码见 [eval_co3d_hydrant_camera_pose.py](/e:/vgg_xzy/vggt/tools/eval_co3d_hydrant_camera_pose.py:100)。

---

**八、单对图像的误差怎么算**

对每一对图像，计算两个误差。

1. 旋转误差 `R_error`

比较预测相对旋转和真值相对旋转之间的角度差，单位是度：

```text
R_error = angle(R_pred, R_gt)
```

它表示模型预测相机朝向变化有多准确。

2. 平移方向误差 `t_error`

比较预测相对平移方向和真值相对平移方向之间的夹角：

```text
t_error = angle(t_pred, t_gt)
```

这里比较的是方向，而不是长度，因为单目位姿存在尺度不确定性。

脚本还处理了平移方向的正负歧义，将 `t` 与 `-t` 看作等价方向之一。

3. 最终位姿误差 `pose_error`

每对图像的最终误差定义为：

```text
pose_error = max(R_error, t_error)
```

也就是说，旋转和平移方向只要有一个很差，该图像对就认为位姿估计较差。

---

**九、AUC@3 / AUC@5 / AUC@15 / AUC@30 表示什么**

Camera Pose 实验使用：

```text
AUC@3
AUC@5
AUC@15
AUC@30
```

含义是：统计所有图像对的 `pose_error` 分布，在给定角度阈值以内的累计曲线面积。

直观理解：

- `AUC@3`：非常严格，要求位姿极准；
- `AUC@5`：严格；
- `AUC@15`：中等；
- `AUC@30`：较宽松，反映大体运动是否预测正确。

指标越高越好。

此外还报告：

```text
median pose
median R
median t
R acc@5
t acc@5
```

其中：

- `median R`：旋转误差中位数；
- `median t`：平移方向误差中位数；
- `R acc@5`：旋转误差小于 `5 deg` 的比例；
- `t acc@5`：平移方向误差小于 `5 deg` 的比例。

---

**十、实验结果该如何理解**

Camera Pose 结果如下：

| 组别 | AUC@30 | AUC@15 | AUC@5 | median pose | median R | median t |
|---|---:|---:|---:|---:|---:|---:|
| 开头静态 | 9.85 | 2.22 | 0.00 | 38.79 deg | 0.07 deg | 38.79 deg |
| 中段大运动 A | 69.48 | 39.41 | 2.67 | 9.05 deg | 1.47 deg | 9.05 deg |
| 中段大运动 B | 72.44 | 57.78 | 27.56 | 5.32 deg | 1.10 deg | 5.32 deg |

最值得讲的不是简单地说“B 最好”，而是误差由什么造成的。

**静态组**

静态组的 `median R = 0.07 deg`，说明模型对相机旋转判断非常准确。

但它的：

```text
baseline = 0.0016 m
median t = 38.79 deg
```

几乎没有真实平移时，所谓“平移方向”本身就不稳定。真实移动可能只有毫米级，极小的噪声都会让方向角发生很大变化。

所以：

> 静态组 AUC 很低，并不能说明 VGGT 看不懂图像，而是该片段不适合用平移方向指标评价位姿。

**中段大运动 A**

这一段有明显的平移和旋转，因此评测条件更加合理：

```text
median R = 1.47 deg
median t = 9.05 deg
AUC@30 = 69.48
```

说明 VGGT 能较好预测整体运动，但平移方向精度还不够稳定，严格的 `AUC@5` 很低。

**中段大运动 B**

这一段表现最好：

```text
median R = 1.10 deg
median t = 5.32 deg
AUC@30 = 72.44
AUC@5 = 27.56
```

可以看出：

- 旋转估计一直都比较可靠；
- 整体位姿性能的主要瓶颈是平移方向估计；
- 有明显运动的图像序列，才真正体现 VGGT 的直接相机位姿恢复能力。

---

**十一、它和 Image Matching 位姿实验的区别**

同一个 EuRoC 报告里还有一张 `Image Matching` 表，容易混淆。

| 项目 | Camera Pose Estimation | Image Matching 中的 Pose |
|---|---|---|
| 位姿来源 | VGGT `camera_head` 直接预测 | 匹配点经过 Essential Matrix 与 `recoverPose` 恢复 |
| 是否使用 ALIKED | 否 | 是 |
| 是否使用 VGGT tracking head | 否 | 是 |
| 是否使用 RANSAC | 否 | 是 |
| 评测 pair 数量 | 10 帧全部两两组合，共 45 对 | 相邻选中帧，共 9 对 |
| 关注问题 | 模型能否直接理解相机运动 | 匹配点能否支撑几何恢复 |

这也解释了一个看似矛盾的现象：

静态组中：

```text
Camera Pose AUC@30 = 9.85
Matching GT epipolar ratio@2px = 99.97%
```

这两者并不冲突。

- 匹配点几乎完全正确，因为图像变化很小；
- 但相机几乎没有平移，因此平移方向无法稳定评估；
- 所以匹配很好，位姿 AUC 仍然可能很低。

---

**十二、CO3D 实验在这里起什么作用**

在 EuRoC 之前，项目中已经有一套 CO3D `hydrant` 的 Camera Pose 脚本：

[eval_co3d_hydrant_camera_pose.py](/e:/vgg_xzy/vggt/tools/eval_co3d_hydrant_camera_pose.py:1)

它更加接近 VGGT 官方 CO3D 位姿评测流程：

```text
从一个物体序列中采样 N 帧
 -> VGGT 直接预测每帧外参
 -> 所有帧两两组合
 -> 计算旋转误差和平移方向误差
 -> 汇总 AUC@30/15/5/3
```

EuRoC 实验本质上复用了这套评测指标，但加入了：

- 真实机器人运动轨迹；
- 时间戳对齐；
- 镜头去畸变；
- 可控的静态/大运动片段对比；
- 同时进行 matching 对照分析。

因此，EuRoC 三组实验更适合拿来解释：

> 位姿估计性能不仅取决于模型，也强烈取决于场景中是否存在足够可观测的相机运动。

---

**一句话概括实验流程**

```text
从 EuRoC V103 选择不同运动强度的 10 帧序列
 -> 根据时间戳和标定获得真值相机外参
 -> 图像去畸变后送入 VGGT
 -> camera_head 直接预测每帧外参
 -> 对 10 帧构造 45 个相对位姿对
 -> 计算旋转误差与平移方向误差
 -> 用 pose_error=max(R_error, t_error) 汇总 AUC
 -> 分析静态退化和大运动条件下的位姿恢复能力
```

这个实验最核心的结论是：**VGGT 对旋转估计非常稳定，直接位姿估计的主要困难集中在平移方向；当真实基线接近零时，平移方向指标会退化，因此必须结合运动幅度解释 AUC。**