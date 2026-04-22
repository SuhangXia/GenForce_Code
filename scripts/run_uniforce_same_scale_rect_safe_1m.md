# `run_uniforce_same_scale_rect_safe_1m.sh` Usage

这个脚本是 `run_uniforce_same_scale_rect_safe_499k.sh` 的 1m 包装版。

它默认只改两项：

- `OUTPUT_ROOT=/home/suhang/datasets/uniforce_same_scale_rect_safe_1m_short16_all18`
- `EPISODES_PER_INDENTER=140`

其余配置和 `499k` 版保持一致，包括：

- `RENDER_SAMPLES=16`
- `MAX_PHYSICS_WORKERS=3`
- `MAX_MESHING_WORKERS=4`
- `MAX_RENDER_WORKERS=12`
- `AUTO_BALANCE_PIPELINE=1`
- `PHYSICS_NPZ_CLEANUP=delete_after_scale_complete`

## 1. 图数说明

当前固定结构是：

- `18` 个压头
- `18` frames / episode
- `22` marker / frame

所以每增加 `1` 个 episode / indenter，就会增加：

```bash
18 * 18 * 22 = 7128
```

张图。

因此：

- `140` episodes / indenter => `997,920` 张图
- `141` episodes / indenter => `1,005,048` 张图

脚本默认取 `140`，因为它更接近 `1,000,000`。

## 2. 直接运行

```bash
bash /home/suhang/projects/test_code/GenForce_Code/scripts/run_uniforce_same_scale_rect_safe_1m.sh
```

## 3. 如果你想强行跑到 100w 以上

```bash
EPISODES_PER_INDENTER=141 \
bash /home/suhang/projects/test_code/GenForce_Code/scripts/run_uniforce_same_scale_rect_safe_1m.sh
```

## 4. 其他参数覆盖

和 `499k` 版完全一样，也支持临时覆盖：

```bash
OUTPUT_ROOT=/home/suhang/datasets/my_same_scale_1m \
MAX_PHYSICS_WORKERS=4 \
MAX_MESHING_WORKERS=4 \
MAX_RENDER_WORKERS=12 \
RENDER_BACKLOG_HIGH_WATERMARK=24 \
RENDER_BACKLOG_LOW_WATERMARK=12 \
bash /home/suhang/projects/test_code/GenForce_Code/scripts/run_uniforce_same_scale_rect_safe_1m.sh
```
