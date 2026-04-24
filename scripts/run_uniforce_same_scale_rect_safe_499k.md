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
source "$(conda info --base)/etc/profile.d/conda.sh"
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
RENDER_DEVICE=gpu
RENDER_GPU_BACKEND=auto
RENDER_SAMPLES=32
MAX_PHYSICS_WORKERS=1
MAX_MESHING_WORKERS=4
MAX_RENDER_WORKERS=1
AUTO_BALANCE_PIPELINE=1
PHYSICS_NPZ_CLEANUP=keep
RESUME=0
KEEP_WORK=1
```

其中：

- `RENDER_SAMPLES` 已固定为 `32`
- `physics` 本身一直走 GPU；这里只是把 `render` 默认也切到 GPU
- 只会读取 `sim/marker/marker_pattern/4_3` 下的 marker 图
- `marker / frame` 会按该目录实际图片数量自动统计，当前是 `22`
- 会开启 render backlog 驱动的自动限流
- 自动限流不会动态修改 worker 数；它做的是在 render backlog 过高时暂停新的 meshing / physics 提交
- 默认会保留 `_work` 和 physics `npz`，优先保证可续跑、可排查、不会自动删除已生成内容
- 运行日志里会额外输出全 `18` 个压头的总进度，而不是只看当前压头
- 因为 physics 和 render 共享 GPU，默认把 `MAX_PHYSICS_WORKERS` 和 `MAX_RENDER_WORKERS` 设得更保守
- `RESUME=1` 时会复用已有 `_work/run_*` 继续跑，而不是要求输出目录必须为空
- 如果你明确想回收空间，可以手动传 `KEEP_WORK=0` 或 `PHYSICS_NPZ_CLEANUP=delete_after_scale_complete`

## 3. 直接运行

```bash
cd /media/zhuochen/data/ssd/suhang/GenForce_Code
bash scripts/run_uniforce_same_scale_rect_safe_499k.sh
```

## 4. 临时覆盖输出路径

```bash
OUTPUT_ROOT=/home/suhang/datasets/my_same_scale_run \
bash /media/zhuochen/data/ssd/suhang/GenForce_Code/scripts/run_uniforce_same_scale_rect_safe_499k.sh
```

## 4.1 断点续跑

如果上次已经在同一个 `OUTPUT_ROOT` 下生成出了 `_work`，可以这样继续：

```bash
OUTPUT_ROOT=/home/suhang/datasets/my_same_scale_run \
RESUME=1 \
bash /media/zhuochen/data/ssd/suhang/GenForce_Code/scripts/run_uniforce_same_scale_rect_safe_499k.sh
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
bash /media/zhuochen/data/ssd/suhang/GenForce_Code/scripts/run_uniforce_same_scale_rect_safe_499k.sh
```

说明：

- `RENDER_BACKLOG_HIGH_WATERMARK=0` / `LOW=0` 表示交给底层生成器自动推导
- `AUTO_BALANCE_PIPELINE=0` 可以关闭自动限流
- `PHYSICS_NPZ_CLEANUP=keep` 可以关闭 NPZ 清理，但一般不建议
- `CONDA_ENV_NAME=genforce` 是默认环境名
- 如果自动探测不到 conda，可以手动传 `CONDA_SH=/your/miniconda3/etc/profile.d/conda.sh`
- 如果是 NVIDIA 卡，通常可以试 `RENDER_GPU_BACKEND=optix` 或 `RENDER_GPU_BACKEND=cuda`
- 如果 `physics` 和 `render` 同时走同一张 GPU，一般不建议把 `MAX_RENDER_WORKERS` 设得太大
- 顶层 `resume` 只会复用同一个 `OUTPUT_ROOT` 下已有的 `_work` 和已合并 `episode_*`

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
