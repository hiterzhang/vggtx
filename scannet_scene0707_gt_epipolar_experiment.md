# ScanNet-1500 scene0707_00 图像匹配真值评估实验总结

日期：2026-05-25  
工程目录：`/home/zzh/vgg/vggt`  
实验脚本：`tools/eval_scannet_scene_gt_epipolar.py`  
实验输出：`outputs/matching_scannet_gt_epipolar_scene0707`

## 1. 实验目标

这次实验针对 ScanNet-1500 中的单个场景 `scene0707_00`，用 VGGT 做 image matching，并用 ScanNet 提供的真值相机参数评估匹配结果。

实验同时看两个指标：

1. `pose AUC`：匹配点能否恢复相机相对位姿。
2. `GT epipolar inlier ratio`：VGGT 输出的每一对匹配点本身是否符合 ScanNet 真值几何。

这比单纯用 RANSAC 自估计 Fundamental Matrix 更可靠，因为红线和绿线不再由匹配点自己投票决定，而是由数据集真值相机位姿决定。

## 2. 数据来源

使用本地已下载的 ScanNet-1500 数据：

```bash
data/scannet1500/scannet1500/
```

本实验读取的配对文件：

```bash
data/scannet1500/scannet1500/pairs_calibrated.txt
```

`scene0707_00` 在 `pairs_calibrated.txt` 中共有 15 对图像：

| idx | image0 | image1 |
|---:|---|---|
| 0000 | `scene0707_00/color/15.jpg` | `scene0707_00/color/585.jpg` |
| 0001 | `scene0707_00/color/45.jpg` | `scene0707_00/color/105.jpg` |
| 0002 | `scene0707_00/color/45.jpg` | `scene0707_00/color/690.jpg` |
| 0003 | `scene0707_00/color/60.jpg` | `scene0707_00/color/585.jpg` |
| 0004 | `scene0707_00/color/90.jpg` | `scene0707_00/color/660.jpg` |
| 0005 | `scene0707_00/color/105.jpg` | `scene0707_00/color/600.jpg` |
| 0006 | `scene0707_00/color/135.jpg` | `scene0707_00/color/165.jpg` |
| 0007 | `scene0707_00/color/150.jpg` | `scene0707_00/color/660.jpg` |
| 0008 | `scene0707_00/color/150.jpg` | `scene0707_00/color/690.jpg` |
| 0009 | `scene0707_00/color/165.jpg` | `scene0707_00/color/660.jpg` |
| 0010 | `scene0707_00/color/375.jpg` | `scene0707_00/color/450.jpg` |
| 0011 | `scene0707_00/color/510.jpg` | `scene0707_00/color/540.jpg` |
| 0012 | `scene0707_00/color/525.jpg` | `scene0707_00/color/540.jpg` |
| 0013 | `scene0707_00/color/585.jpg` | `scene0707_00/color/630.jpg` |
| 0014 | `scene0707_00/color/765.jpg` | `scene0707_00/color/780.jpg` |

## 3. 运行命令

进入工程并激活环境：

```bash
cd /home/zzh/vgg/vggt
source /home/zzh/anaconda3/bin/activate vggt
```

完整保存可视化、npz 和 txt 的命令：

```bash
python tools/eval_scannet_scene_gt_epipolar.py \
  --scene scene0707_00 \
  --out_root outputs/matching_scannet_gt_epipolar_scene0707 \
  --tag scene0707_00 \
  --max_keypoints 5000 \
  --track_chunk 1024 \
  --gt_epipolar_threshold_px 2.0 \
  --max_draw 300
```

如果以后只想节省空间，只保存最终 CSV 和 JSON，可以加：

```bash
--no-save_pair_outputs
```

完整轻量命令示例：

```bash
python tools/eval_scannet_scene_gt_epipolar.py \
  --scene scene0707_00 \
  --out_root outputs/matching_scannet_gt_epipolar_scene0707_light \
  --tag scene0707_00 \
  --max_keypoints 5000 \
  --track_chunk 1024 \
  --gt_epipolar_threshold_px 2.0 \
  --max_draw 300 \
  --no-save_pair_outputs
```

## 4. 实验流程

1. 从 `pairs_calibrated.txt` 中筛选 `scene0707_00` 的全部图像对。
2. 对每张图像执行 VGGT 的预处理，本次使用 `crop`。
3. 在第一张图上用 `ALIKED` 提取 query keypoints，最多 5000 个点。
4. 使用 VGGT 的 `track_head` 从 image0 跟踪到 image1，得到匹配点。
5. 按 `vis_threshold=0.5` 和边界条件过滤无效匹配。
6. 用匹配点通过 RANSAC 估计相对位姿，再和 ScanNet 真值相对位姿比较，得到 pose error。
7. 用 ScanNet 真值内参和相对位姿构造真值 Fundamental Matrix。
8. 对每个匹配点计算 Sampson epipolar error。
9. 以 `2.0 px` 为阈值判断真值几何内点，并画红绿线可视化。
10. 汇总 `pose AUC@5/10/20` 和 `GT epipolar inlier ratio`。

## 5. 关键参数

| 参数 | 值 | 说明 |
|---|---:|---|
| `max_keypoints` | 5000 | ALIKED 最多提取 5000 个 query keypoints |
| `aliked_threshold` | 0.005 | ALIKED 检测阈值 |
| `vis_threshold` | 0.5 | VGGT track 可见性过滤阈值 |
| `conf_threshold` | 0.0 | 未额外使用 confidence 过滤 |
| `track_chunk` | 1024 | 分块跟踪，降低显存压力 |
| `eval_resize_min` | 480 | 评估时将图像短边缩放到 480 |
| `pose_ransac_px` | 0.5 | 位姿估计 RANSAC 阈值 |
| `ransac_confidence` | 0.99999 | RANSAC 置信度 |
| `ransac_iters` | 10000 | RANSAC 最大迭代次数 |
| `gt_epipolar_threshold_px` | 2.0 | 真值 epipolar 内点阈值 |
| `preprocess` | crop | VGGT 图像预处理方式 |

## 6. 指标解释

### 6.1 Pose AUC

`pose AUC@5/10/20` 衡量由匹配点恢复出来的相机相对位姿有多准。

流程是：

1. 用 VGGT 匹配点估计 Essential Matrix。
2. 从 Essential Matrix 恢复相对旋转和平移方向。
3. 和 ScanNet 真值相对位姿比较，得到旋转误差和平移方向误差。
4. 取二者较大的那个作为该图像对的 `pose_error_deg`。
5. 统计 pose error 在 5、10、20 度以内的误差曲线面积。

数值越高越好。AUC@5 更严格，AUC@20 更宽松。

### 6.2 GT epipolar inlier ratio

`GT epipolar inlier ratio` 衡量匹配点本身是否符合真值几何。

脚本使用 ScanNet 给出的真值相对位姿和相机内参构造：

```text
F_gt = K1^-T [t]_x R K0^-1
```

然后对每一对匹配点计算 Sampson epipolar error。若误差 `<= 2.0 px`，则认为是真值几何内点。

这个指标不依赖 RANSAC 自估计结果，所以更适合判断“这些匹配线本身是不是真的几何正确”。

### 6.3 红线和绿线

本实验输出的 `matches_scannet_gt_epipolar.png` 中：

| 颜色 | 含义 |
|---|---|
| 绿色 | 该匹配点的真值 Sampson epipolar error `<= 2.0 px` |
| 红色 | 该匹配点的真值 Sampson epipolar error `> 2.0 px` |

注意：这里的红绿线不是 RANSAC 自估计出来的内点和外点，而是由 ScanNet 真值位姿判断的。

## 7. 总体结果

| 指标 | 结果 |
|---|---:|
| 场景 | `scene0707_00` |
| 图像对数量 | 15 |
| Pose AUC@5 | 44.07 |
| Pose AUC@10 | 61.56 |
| Pose AUC@20 | 73.33 |
| Median pose error | 2.15 deg |
| 总匹配数 | 60349 |
| GT epipolar 内点数，2px | 27280 |
| 全局 GT epipolar inlier ratio，2px | 45.20% |
| 逐 pair 平均 GT epipolar inlier ratio，2px | 41.79% |
| 平均 median GT epipolar error | 2.93 px |
| 平均 mean GT epipolar error | 9.68 px |

pose error 阈值统计：

| 阈值 | 达标 pair 数 | 比例 |
|---:|---:|---:|
| <= 5 deg | 11 / 15 | 73.33% |
| <= 10 deg | 12 / 15 | 80.00% |
| <= 20 deg | 13 / 15 | 86.67% |

GT epipolar error 阈值统计：

| 阈值 | 内点数 | 全局比例 |
|---:|---:|---:|
| <= 0.5 px | 6667 / 60349 | 11.05% |
| <= 1 px | 14045 / 60349 | 23.27% |
| <= 2 px | 27280 / 60349 | 45.20% |
| <= 3 px | 36271 / 60349 | 60.10% |
| <= 5 px | 45545 / 60349 | 75.47% |
| <= 10 px | 51029 / 60349 | 84.56% |

## 8. 逐对详细结果

| idx | image0 -> image1 | query | matches | pose err deg | rot err | trans err | pose inliers | GT inliers @2px | GT ratio | median epi px | mean epi px |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0000 | 15 -> 585 | 4666 | 4654 | 1.60 | 0.32 | 1.60 | 1517 | 3637 | 78.15% | 1.00 | 1.22 |
| 0001 | 45 -> 105 | 4632 | 3641 | 2.31 | 1.76 | 2.31 | 996 | 543 | 14.91% | 4.46 | 22.27 |
| 0002 | 45 -> 690 | 4632 | 1480 | 7.35 | 2.93 | 7.35 | 606 | 54 | 3.65% | 5.33 | 17.17 |
| 0003 | 60 -> 585 | 4646 | 4646 | 1.57 | 1.51 | 1.57 | 3822 | 3870 | 83.30% | 1.46 | 1.39 |
| 0004 | 90 -> 660 | 4665 | 4650 | 1.58 | 1.58 | 1.43 | 1896 | 1842 | 39.61% | 3.10 | 5.26 |
| 0005 | 105 -> 600 | 4668 | 4668 | 1.86 | 1.86 | 0.70 | 2612 | 2191 | 46.94% | 2.08 | 2.41 |
| 0006 | 135 -> 165 | 4631 | 4546 | 17.33 | 0.74 | 17.33 | 1272 | 1469 | 32.31% | 3.50 | 6.86 |
| 0007 | 150 -> 660 | 4589 | 4416 | 2.15 | 2.15 | 1.52 | 1977 | 2349 | 53.19% | 1.88 | 2.19 |
| 0008 | 150 -> 690 | 4589 | 4558 | 4.08 | 2.89 | 4.08 | 1679 | 2030 | 44.54% | 2.33 | 3.13 |
| 0009 | 165 -> 660 | 4644 | 4644 | 1.55 | 1.55 | 0.71 | 2667 | 3086 | 66.45% | 1.32 | 2.42 |
| 0010 | 375 -> 450 | 4665 | 4114 | 1.94 | 1.46 | 1.94 | 1891 | 2007 | 48.78% | 2.14 | 21.57 |
| 0011 | 510 -> 540 | 4684 | 3150 | 2.03 | 1.74 | 2.03 | 1483 | 744 | 23.62% | 3.25 | 8.64 |
| 0012 | 525 -> 540 | 4682 | 3954 | 27.81 | 3.50 | 27.81 | 513 | 1686 | 42.64% | 2.20 | 9.50 |
| 0013 | 585 -> 630 | 4662 | 2929 | 3.32 | 0.19 | 3.32 | 1296 | 697 | 23.80% | 4.05 | 27.88 |
| 0014 | 765 -> 780 | 4609 | 4299 | 72.14 | 5.81 | 72.14 | 784 | 1075 | 25.01% | 5.91 | 13.29 |

## 9. 结果分析

### 9.1 位姿恢复整体较好

15 对图像里有 11 对的 pose error 小于 5 度，12 对小于 10 度，13 对小于 20 度。`Pose AUC@20 = 73.33`，说明在 `scene0707_00` 这个小场景上，VGGT 的匹配点通常足够支撑相机相对位姿恢复。

最好的几对包括：

| idx | 图像对 | pose error | GT ratio @2px |
|---:|---|---:|---:|
| 0009 | 165 -> 660 | 1.55 deg | 66.45% |
| 0003 | 60 -> 585 | 1.57 deg | 83.30% |
| 0004 | 90 -> 660 | 1.58 deg | 39.61% |
| 0000 | 15 -> 585 | 1.60 deg | 78.15% |
| 0005 | 105 -> 600 | 1.86 deg | 46.94% |

其中 `0003` 和 `0000` 不仅 pose error 小，GT epipolar ratio 也很高，可视化里应该能看到大量绿线。

### 9.2 GT epipolar ratio 比 pose AUC 更严格

虽然 pose AUC 表现不错，但全局 `GT epipolar inlier ratio@2px` 只有 45.20%。这不是矛盾，因为两个指标关注点不同：

| 指标 | 关注什么 | 特点 |
|---|---|---|
| pose AUC | 匹配点能否支撑恢复相机位姿 | 只需要足够多的高质量内点，RANSAC 可以丢掉外点 |
| GT epipolar ratio | 每条匹配线是否符合真值几何 | 对所有保留下来的匹配点逐一打分，更严格 |

所以会出现这样的情况：某些 pair 的 pose error 很小，但红线仍然不少。这说明正确匹配点足够恢复位姿，但输出的全部匹配中仍混有不少几何误差较大的点。

### 9.3 2px 阈值很严格

在 2px 阈值下，全局 GT epipolar ratio 是 45.20%。如果放宽阈值：

| 阈值 | GT ratio |
|---:|---:|
| 2 px | 45.20% |
| 3 px | 60.10% |
| 5 px | 75.47% |
| 10 px | 84.56% |

这说明很多红线并不是完全离谱，而是误差落在 2 到 10 像素之间。对于严格几何评估，2px 更能反映高精度匹配质量；对于可视化观察，5px 或 10px 会看起来宽松很多。

### 9.4 失败或偏弱 pair

最明显的问题 pair 是：

| idx | 图像对 | pose error | GT ratio @2px | 现象 |
|---:|---|---:|---:|---|
| 0014 | 765 -> 780 | 72.14 deg | 25.01% | 位姿恢复失败，平移方向误差很大 |
| 0012 | 525 -> 540 | 27.81 deg | 42.64% | 匹配点不少，但位姿估计结果不好 |
| 0006 | 135 -> 165 | 17.33 deg | 32.31% | 平移方向误差偏大 |
| 0002 | 45 -> 690 | 7.35 deg | 3.65% | 真值 epipolar 内点极少，匹配几何质量很差 |
| 0001 | 45 -> 105 | 2.31 deg | 14.91% | 位姿可恢复，但大量匹配不满足 2px 真值几何 |

特别是 `0014`，pose error 达到 72.14 度，应视为该场景里的一次明显失败。

`0002` 的情况也值得注意：GT epipolar ratio 只有 3.65%，但 pose error 是 7.35 度，没有像 `0014` 那样完全崩掉。这说明 RANSAC 可能从少量正确匹配中仍找到了一组相对合理的位姿，但整体匹配点集合质量很低。

## 10. 与之前 CO3D hydrant 真值实验的区别

之前 CO3D hydrant 的跳帧相邻匹配实验中，GT epipolar inlier ratio 非常高，整体接近 99.93%。这次 ScanNet `scene0707_00` 在 2px 阈值下只有 45.20%。

主要原因可能包括：

1. ScanNet 是室内 RGB-D 扫描场景，遮挡、重复纹理、弱纹理区域更多。
2. ScanNet-1500 的图像对视角变化可能更复杂，并不一定是相邻帧。
3. VGGT 输出的是稠密倾向的 track 匹配，保留了很多可见性高但几何精度未必达到 2px 的点。
4. 本实验对每条匹配都用真值几何严格检查，而不是只看 RANSAC 内点。
5. 2px Sampson error 阈值本身比较严格，放宽到 5px 后全局比例提升到 75.47%。

## 11. 输出文件说明

输出目录：

```bash
outputs/matching_scannet_gt_epipolar_scene0707/
```

目录大小约 12 MB，共 47 个文件。

核心汇总文件：

```bash
outputs/matching_scannet_gt_epipolar_scene0707/scene0707_00_metrics.json
outputs/matching_scannet_gt_epipolar_scene0707/scene0707_00_results.csv
```

每个 pair 的输出目录形如：

```bash
outputs/matching_scannet_gt_epipolar_scene0707/scene0707_00_0000/
```

其中包含：

| 文件 | 用途 |
|---|---|
| `matches_scannet_gt_epipolar.png` | 红绿线匹配可视化 |
| `matches_scannet_gt_epipolar.npz` | 原始匹配点、分数、GT epipolar error、内外点 mask |
| `summary_scannet_gt_epipolar.txt` | 单个 pair 的文字汇总 |

## 12. 结论

在 ScanNet-1500 的 `scene0707_00` 场景上，VGGT + ALIKED 的匹配结果能够较好支撑相机位姿恢复：`Pose AUC@5/10/20 = 44.07 / 61.56 / 73.33`，15 对中有 11 对 pose error 小于 5 度。

但从真值 epipolar 几何逐点检查来看，匹配点精度没有 CO3D hydrant 那组轻量实验那么高：2px 阈值下全局 GT epipolar inlier ratio 为 45.20%。这说明在 ScanNet 室内场景中，VGGT 的匹配点有足够多的正确内点用于位姿恢复，但输出的完整匹配集合中仍包含较多几何误差偏大的点。

因此，如果简报中要评价这个实验，可以这样概括：

> 在 `scene0707_00` 上，VGGT 的匹配具备较强的相机位姿恢复能力，但严格真值 epipolar 检查显示，逐点匹配精度仍受场景复杂度、视角变化和阈值设置影响。pose AUC 和 GT epipolar ratio 应该一起看，前者反映能否恢复位姿，后者反映匹配点集合本身的几何纯度。
