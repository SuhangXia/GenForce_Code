# `run_uniforce_same_scale_rect_safe_499k.sh` Usage

这个脚本用于生成：

- same-scale
- `4:3` rectgel
- 短边 `16mm`
- `18` 个压头全覆盖
- 只使用 `sim/marker/marker_pattern/4_3` 里的 marker
- `640x480`
- safe-depth 只对指定的 5 个压头生效

默认数据规模约为：

- 当前 marker 数量：`22`
- `70` episodes / indenter
- `18` frames / episode
- `22` marker / frame
- 总图数约 `498,960`

## 1. 先激活环境

```bash
source /home/suhang/anaconda3/etc/profile.d/conda.sh
conda activate genforce
```

## 2. 当前默认配置

这个脚本现在默认等价于：

```bash
OUTPUT_ROOT=/home/suhang/datasets/uniforce_same_scale_rect_safe_499k_short16_all18
SHORT_EDGE_MM=16.0
EPISODES_PER_INDENTER=70
DEPTH_MIN_MM=0.5
REQUESTED_DEPTH_MAX_MM=2.0
RENDER_DEVICE=cpu
RENDER_SAMPLES=16
MAX_PHYSICS_WORKERS=3
MAX_MESHING_WORKERS=4
MAX_RENDER_WORKERS=12
AUTO_BALANCE_PIPELINE=1
PHYSICS_NPZ_CLEANUP=delete_after_scale_complete
```

其中：

- `RENDER_SAMPLES` 已固定为 `16`
- 只会读取 `sim/marker/marker_pattern/4_3` 下的 marker 图
- `marker / frame` 会按该目录实际图片数量自动统计，当前是 `22`
- 会开启 render backlog 驱动的自动限流
- 会在 scale 完成后删除 physics `npz`，避免缓存把磁盘吃满
- 运行日志里会额外输出全 `18` 个压头的总进度，而不是只看当前压头

## 3. 直接运行

```bash
bash /home/suhang/projects/test_code/GenForce_Code/scripts/run_uniforce_same_scale_rect_safe_499k.sh
```

## 4. 临时覆盖输出路径

```bash
OUTPUT_ROOT=/home/suhang/datasets/my_same_scale_run \
bash /home/suhang/projects/test_code/GenForce_Code/scripts/run_uniforce_same_scale_rect_safe_499k.sh
```

## 5. 临时覆盖并行度或限流阈值

如果你想在不改脚本的前提下临时调参，可以这样跑：

```bash
OUTPUT_ROOT=/home/suhang/datasets/my_same_scale_run \
MAX_PHYSICS_WORKERS=4 \
MAX_MESHING_WORKERS=4 \
MAX_RENDER_WORKERS=12 \
RENDER_BACKLOG_HIGH_WATERMARK=24 \
RENDER_BACKLOG_LOW_WATERMARK=12 \
bash /home/suhang/projects/test_code/GenForce_Code/scripts/run_uniforce_same_scale_rect_safe_499k.sh
```

说明：

- `RENDER_BACKLOG_HIGH_WATERMARK=0` / `LOW=0` 表示交给底层生成器自动推导
- `AUTO_BALANCE_PIPELINE=0` 可以关闭自动限流
- `PHYSICS_NPZ_CLEANUP=keep` 可以关闭 NPZ 清理，但一般不建议

## 6. 运行时日志

脚本运行时会：

- 在终端打印总任务数、总 render task 和总图数
- 输出底层 `Render progress | ...`
- 输出 `Pipeline auto-balance | ...`
- 输出 `UniForce overall progress | ...`

其中 `UniForce overall progress | ...` 是按全 `18` 个压头累计后的总进度。

完整日志会写到：

```bash
${OUTPUT_ROOT}.run.log
```

例如如果：

```bash
OUTPUT_ROOT=/home/suhang/datasets/my_same_scale_run
```

那么日志文件就是：

```bash
/home/suhang/datasets/my_same_scale_run.run.log
```
