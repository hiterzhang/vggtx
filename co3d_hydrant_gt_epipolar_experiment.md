# CO3Dv2 Hydrant 有真值 Epipolar Matching 实验总结

## 1. 实验目标

本实验用当前本地已下载的 CO3Dv2 `hydrant` 数据，测试 VGGT 在相邻帧上的 image matching 能力，并进一步使用 CO3D 提供的真值相机 pose 评估匹配点的真实 epipolar error。

前一版可视化中，红绿线来自匹配点自身估计出的 Fundamental Matrix 和 RANSAC。这个判断是常用的几何一致性诊断，但它不是数据集真值。本实验改为：

```text
VGGT 匹配点
→ 读取 CO3D frame_annotations.jgz 中的真值 R/T/内参
→ 构造真值 Fundamental Matrix
→ 计算每个匹配点的真实 Sampson epipolar error
→ 按真值误差重新判断红绿线
```

因此，本实验比之前的自估计 RANSAC 红绿线更适合评价匹配点是否符合 CO3D 真值几何。

## 2. 数据与选帧

数据位置：

```text
/home/zzh/vgg/vggt/data/hydrant
```

真值标注文件：

```text
data/hydrant/frame_annotations.jgz
data/hydrant/sequence_annotations.jgz
```

其中：

```text
frame_annotations.jgz
保存每一帧的 image path、R、T、focal_length、principal_point 等相机参数。

sequence_annotations.jgz
保存 sequence 级信息，例如 viewpoint_quality_score。
```

本次实验使用两个有效 sequence：

```text
167_18184_34441
411_56064_108483
```

`605_94563_187702` 没有纳入本次有真值评估，因为它的：

```text
viewpoint_quality_score = nan
```

官方 CO3D 预处理脚本会保留：

```text
viewpoint_quality_score > 0.5
```

所以跳过该 sequence 更接近官方质量过滤口径。

每个有效 sequence 选取 10 张跳帧图片：

```text
1, 3, 5, 7, 9, 11, 13, 15, 17, 19
```

抽出的图片目录：

```text
data/hydrant_stride_1_2_10f_selected
```

图片数量：

```text
20 张
```

目录大小：

```text
2.6M
```

## 3. 实验流程

### 3.1 抽取测试图片

抽取命令：

```bash
cd /home/zzh/vgg/vggt

out="data/hydrant_stride_1_2_10f_selected"
mkdir -p "$out"

for seq in 167_18184_34441 411_56064_108483; do
  for n in 1 3 5 7 9 11 13 15 17 19; do
    printf -v frame "frame%06d.jpg" "$n"
    cp "data/hydrant/$seq/images/$frame" "$out/${seq}_${frame}"
  done
done
```

### 3.2 相邻帧 VGGT image matching

脚本：

```text
tools/match_hydrant_stride_adjacent.py
```

运行命令：

```bash
cd /home/zzh/vgg/vggt
source /home/zzh/anaconda3/bin/activate vggt

python tools/match_hydrant_stride_adjacent.py \
  --selected_dir data/hydrant_stride_1_2_10f_selected \
  --out_root outputs/matching/hydrant_stride_adjacent \
  --max_points 512 \
  --max_draw 200 \
  --overview_max_draw_per_pair 25 \
  --keypoints aliked
```

匹配设置：

```text
query keypoints: ALIKED
max_points: 512
preprocess: crop
preprocessed image size: 518 x 518
matching model: VGGT tracking head
```

输出目录：

```text
outputs/matching/hydrant_stride_adjacent
```

该目录包含每个相邻 pair 的：

```text
matches.png
matches.npz
summary.txt
```

以及每个 sequence 的总览图：

```text
outputs/matching/hydrant_stride_adjacent/167_18184_34441/overview_adjacent_matches.png
outputs/matching/hydrant_stride_adjacent/411_56064_108483/overview_adjacent_matches.png
```

注意：这一步的红绿线来自自估计 RANSAC，只用于初步可视化，不是 CO3D 真值判断。

### 3.3 用 CO3D 真值 pose 评估 epipolar error

脚本：

```text
tools/evaluate_hydrant_matches_gt_epipolar.py
```

运行命令：

```bash
cd /home/zzh/vgg/vggt
source /home/zzh/anaconda3/bin/activate vggt

python tools/evaluate_hydrant_matches_gt_epipolar.py \
  --matches_root outputs/matching/hydrant_stride_adjacent \
  --out_root outputs/matching/hydrant_stride_adjacent_gt_pose \
  --threshold_px 2.0 \
  --max_draw 200 \
  --overview_max_draw_per_pair 25
```

输出目录：

```text
outputs/matching/hydrant_stride_adjacent_gt_pose
```

核心输出：

```text
outputs/matching/hydrant_stride_adjacent_gt_pose/summary_gt_pose.csv
outputs/matching/hydrant_stride_adjacent_gt_pose/167_18184_34441/overview_gt_pose_matches.png
outputs/matching/hydrant_stride_adjacent_gt_pose/411_56064_108483/overview_gt_pose_matches.png
```

每个 pair 还保存：

```text
matches_gt_pose.png
matches_gt_pose.npz
summary_gt_pose.txt
```

输出大小：

```text
outputs/matching/hydrant_stride_adjacent_gt_pose: 26M
```

## 4. 真值 Epipolar Error 的计算方法

### 4.1 读取 CO3D 真值相机

每一帧在 `frame_annotations.jgz` 中有：

```text
R
T
focal_length
principal_point
intrinsics_format = ndc_isotropic
```

CO3D / PyTorch3D 相机参数先转换成 OpenCV camera-from-world 外参。

### 4.2 构造真值 Fundamental Matrix

对于两帧 image0 和 image1：

```text
T0 = camera0-from-world
T1 = camera1-from-world
T_1_from_0 = T1 * inverse(T0)
R = T_1_from_0[:3, :3]
t = T_1_from_0[:3, 3]
E = [t]_x R
F = K1^-T E K0^-1
```

因为 VGGT 匹配点是在预处理后的 518x518 坐标里，所以还需要把 CO3D 原图内参经过同样的 resize/crop 仿射变换，得到预处理坐标系下的 `K0` 和 `K1`。

### 4.3 Sampson Epipolar Error

对每个匹配点：

```text
x0 = image0 上的点
x1 = image1 上的匹配点
```

计算 Sampson error：

```text
error = |x1^T F x0| /
        sqrt((F x0)_x^2 + (F x0)_y^2 + (F^T x1)_x^2 + (F^T x1)_y^2)
```

本次使用阈值：

```text
threshold_px = 2.0
```

判断方式：

```text
error <= 2.0 px  → GT inlier，绿色
error > 2.0 px   → GT outlier，红色
```

这和之前的 RANSAC 可视化不同：

```text
旧图红绿线：由匹配点自估计 F + RANSAC 判断
新图红绿线：由 CO3D 真值 pose 构造 F 判断
```

## 5. 总体结果

评估规模：

```text
sequence 数: 2
相邻 pair 数: 18
总匹配点数: 8960
```

总体 GT epipolar 结果：

```text
GT inliers @ 2px: 8954 / 8960
GT inlier ratio: 99.93%
```

不同阈值下的 GT epipolar inlier 比例：

```text
error <= 0.25 px: 6789 / 8960 = 75.77%
error <= 0.50 px: 8526 / 8960 = 95.16%
error <= 1.00 px: 8924 / 8960 = 99.60%
error <= 2.00 px: 8954 / 8960 = 99.93%
```

旧 RANSAC 与 GT 的对比：

```text
RANSAC inliers: 8790 / 8960 = 98.10%
RANSAC vs GT precision: 99.99%
RANSAC vs GT recall: 98.16%
```

解释：

```text
RANSAC 标绿的点几乎都是真值正确的，precision 很高。
但 RANSAC 会漏掉一部分真值正确的点，recall 低于 100%。
因此旧 RANSAC 红线中有一部分其实是真值几何正确的匹配。
```

## 6. 按 Sequence 汇总

| sequence | pairs | matches | GT inliers | GT ratio | RANSAC inliers | RANSAC ratio | 平均 median GT error | 平均 mean GT error | RANSAC vs GT precision | RANSAC vs GT recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 167_18184_34441 | 9 | 4402 | 4397 | 99.89% | 4293 | 97.52% | 0.175 px | 0.213 px | 100.00% | 97.61% |
| 411_56064_108483 | 9 | 4558 | 4557 | 99.98% | 4497 | 98.66% | 0.117 px | 0.151 px | 99.98% | 98.65% |

两个 sequence 的匹配都非常稳定。`411_56064_108483` 的平均 GT error 更低一些，说明在真值 epipolar 几何下，这组匹配整体更准。

## 7. 逐 Pair 详细结果

### 7.1 Sequence 167_18184_34441

| frame0 | frame1 | matches | GT inliers | GT ratio | median GT error | mean GT error | RANSAC inliers | RANSAC ratio | RANSAC vs GT recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 507 | 507 | 100.00% | 0.101 | 0.107 | 507 | 100.00% | 100.00% |
| 3 | 5 | 502 | 502 | 100.00% | 0.152 | 0.182 | 502 | 100.00% | 100.00% |
| 5 | 7 | 496 | 496 | 100.00% | 0.190 | 0.214 | 494 | 99.60% | 99.60% |
| 7 | 9 | 498 | 498 | 100.00% | 0.152 | 0.178 | 497 | 99.80% | 99.80% |
| 9 | 11 | 475 | 474 | 99.79% | 0.177 | 0.224 | 462 | 97.26% | 97.47% |
| 11 | 13 | 490 | 490 | 100.00% | 0.213 | 0.254 | 482 | 98.37% | 98.37% |
| 13 | 15 | 480 | 480 | 100.00% | 0.216 | 0.305 | 440 | 91.67% | 91.67% |
| 15 | 17 | 467 | 463 | 99.14% | 0.203 | 0.255 | 460 | 98.50% | 99.35% |
| 17 | 19 | 487 | 487 | 100.00% | 0.173 | 0.200 | 449 | 92.20% | 92.20% |

### 7.2 Sequence 411_56064_108483

| frame0 | frame1 | matches | GT inliers | GT ratio | median GT error | mean GT error | RANSAC inliers | RANSAC ratio | RANSAC vs GT recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 511 | 511 | 100.00% | 0.055 | 0.100 | 510 | 99.80% | 99.80% |
| 3 | 5 | 508 | 508 | 100.00% | 0.057 | 0.105 | 508 | 100.00% | 100.00% |
| 5 | 7 | 508 | 508 | 100.00% | 0.216 | 0.229 | 508 | 100.00% | 100.00% |
| 7 | 9 | 508 | 508 | 100.00% | 0.099 | 0.126 | 508 | 100.00% | 100.00% |
| 9 | 11 | 507 | 507 | 100.00% | 0.101 | 0.139 | 492 | 97.04% | 97.04% |
| 11 | 13 | 505 | 505 | 100.00% | 0.110 | 0.132 | 505 | 100.00% | 100.00% |
| 13 | 15 | 502 | 501 | 99.80% | 0.167 | 0.191 | 459 | 91.43% | 91.42% |
| 15 | 17 | 504 | 504 | 100.00% | 0.080 | 0.120 | 502 | 99.60% | 99.60% |
| 17 | 19 | 505 | 505 | 100.00% | 0.171 | 0.213 | 505 | 100.00% | 100.00% |

## 8. 411_56064_108483 的 13→15 重新解释

之前在自估计 RANSAC 的 overview 图里，`411_56064_108483` 的 13→15 位置看起来红线很多。用 CO3D 真值 pose 重新判断后，这对其实很好：

```text
matches: 502
GT inliers @ 2px: 501
GT inlier ratio: 99.80%
median GT epipolar error: 0.167 px
mean GT epipolar error: 0.191 px
```

旧 RANSAC 结果：

```text
RANSAC inliers: 459
RANSAC inlier ratio: 91.43%
```

对比可以看出：

```text
RANSAC 漏掉了不少真值几何正确的匹配点。
```

因此，之前 overview 里 13→15 红线多，不代表 VGGT 匹配真的差，而是因为：

```text
1. overview 只画 visibility/scores 最高的少量匹配；
2. 自估计 RANSAC 对这一 pair 较保守；
3. 一部分真实符合 CO3D 真值 epipolar 几何的点被 RANSAC 判成 outlier。
```

使用真值 pose 后，13→15 的图基本变为绿色，说明这对匹配在数据集真值几何下是可靠的。

对应文件：

```text
outputs/matching/hydrant_stride_adjacent_gt_pose/411_56064_108483/pair_06_f000013_f000015/matches_gt_pose.png
outputs/matching/hydrant_stride_adjacent_gt_pose/411_56064_108483/pair_06_f000013_f000015/summary_gt_pose.txt
```

## 9. 输出文件说明

### 9.1 汇总 CSV

```text
outputs/matching/hydrant_stride_adjacent_gt_pose/summary_gt_pose.csv
```

重要列：

```text
sequence
pair
frame0
frame1
matches
gt_inliers
gt_inlier_ratio
median_gt_epipolar_error_px
mean_gt_epipolar_error_px
ransac_inliers
ransac_inlier_ratio
ransac_vs_gt_precision
ransac_vs_gt_recall
matches_gt_pose_png
```

含义：

```text
matches
VGGT 输出并通过基础过滤的匹配数量。

gt_inliers
使用 CO3D 真值 pose 判断，epipolar error <= threshold_px 的匹配数。

gt_inlier_ratio
gt_inliers / matches。

median_gt_epipolar_error_px
该 pair 所有匹配点的真值 epipolar error 中位数。

mean_gt_epipolar_error_px
该 pair 所有匹配点的真值 epipolar error 平均值。

ransac_inliers
旧方法中由匹配点自估计 RANSAC 判断为 inlier 的数量。

ransac_vs_gt_precision
RANSAC 标绿的点中，有多少也被 GT pose 判断为正确。

ransac_vs_gt_recall
GT pose 判断正确的点中，有多少被 RANSAC 找到了。
```

### 9.2 GT 可视化图

总览图：

```text
outputs/matching/hydrant_stride_adjacent_gt_pose/167_18184_34441/overview_gt_pose_matches.png
outputs/matching/hydrant_stride_adjacent_gt_pose/411_56064_108483/overview_gt_pose_matches.png
```

每对相邻帧细节图：

```text
outputs/matching/hydrant_stride_adjacent_gt_pose/<sequence>/pair_XX_*/matches_gt_pose.png
```

颜色解释：

```text
绿色: CO3D 真值 pose 下 epipolar error <= 2 px
红色: CO3D 真值 pose 下 epipolar error > 2 px
```

## 10. 结论

本次 CO3Dv2 hydrant 有真值 epipolar 实验表明：

```text
VGGT 在这两个 hydrant sequence 的相邻帧匹配上非常稳定。
总共 8960 个匹配点中，8954 个在 CO3D 真值几何下 epipolar error <= 2 px。
整体 GT inlier ratio 达到 99.93%。
```

并且：

```text
RANSAC 自估计红绿线适合快速诊断，但会漏掉部分真值正确点。
CO3D 真值 pose 评估更适合作为本实验中的几何正确性判断。
```

对于之前讨论的 `411_56064_108483` 的 13→15：

```text
旧 RANSAC 图看起来红线多；
但 GT pose 评估显示 501/502 个匹配点是真值 epipolar inlier；
因此这对图不是匹配失败，而是 RANSAC 可视化误导了观感。
```

## 11. 局限性

本实验仍然是轻量局部测试，不是完整 CO3Dv2 benchmark：

```text
只测试 hydrant 类别；
只测试两个有效 sequence；
只测试相邻跳帧 pair；
没有覆盖 CO3Dv2 全部类别和完整官方评估划分。
```

另外，epipolar error 是几何约束指标，它验证匹配点是否符合两视图真值几何，但不等价于人工语义级对应点标注。在常规 image matching 评估中，这仍然是非常重要且标准的几何评价方式。
