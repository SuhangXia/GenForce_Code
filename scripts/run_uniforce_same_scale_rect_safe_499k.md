# `run_uniforce_same_scale_rect_safe_499k.sh` Usage

这个脚本用于生成：

- same-scale
- `4:3` rectgel
- 短边 `16mm`
- `18` 个压头全覆盖
- `20` 个 marker
- `640x480`
- safe-depth 只对指定的 5 个压头生效

默认数据规模约为：

- `77` episodes / indenter
- `18` frames / episode
- `20` marker / frame
- 总图数约 `498,960`

## 1. 先激活环境

```bash
source /home/suhang/anaconda3/etc/profile.d/conda.sh
conda activate genforce
```

## 2. 当前推荐配置

如果你要的配置是：

- `physics workers = 3`
- `mesh workers = 4`
- `render workers = 12`

那么先把脚本顶部这三行改成：

```bash
MAX_PHYSICS_WORKERS=3
MAX_MESHING_WORKERS=4
MAX_RENDER_WORKERS=12
```

然后直接运行：

```bash
bash /home/suhang/projects/test_code/GenForce_Code/scripts/run_uniforce_same_scale_rect_safe_499k.sh
```

## 3. 手动指定输出路径

脚本支持手动指定输出路径，不需要改 Python 文件。

例如输出到你自己的目录：

```bash
source /home/suhang/anaconda3/etc/profile.d/conda.sh
conda activate genforce

OUTPUT_ROOT=/home/suhang/datasets/my_same_scale_run \
bash /home/suhang/projects/test_code/GenForce_Code/scripts/run_uniforce_same_scale_rect_safe_499k.sh
```

## 4. 手动指定路径 + 指定并行度

如果你不想直接改 `.sh` 文件，也可以临时覆盖参数：

```bash
source /home/suhang/anaconda3/etc/profile.d/conda.sh
conda activate genforce

OUTPUT_ROOT=/home/suhang/datasets/my_same_scale_run \
MAX_PHYSICS_WORKERS=3 \
MAX_MESHING_WORKERS=4 \
MAX_RENDER_WORKERS=12 \
bash /home/suhang/projects/test_code/GenForce_Code/scripts/run_uniforce_same_scale_rect_safe_499k.sh
```

## 5. 脚本里当前关键默认项

脚本默认目前等价于：

```bash
OUTPUT_ROOT=/home/suhang/datasets/uniforce_same_scale_rect_safe_499k_short16_all18
SHORT_EDGE_MM=16.0
EPISODES_PER_INDENTER=77
DEPTH_MIN_MM=0.5
REQUESTED_DEPTH_MAX_MM=2.0
RENDER_DEVICE=cpu
RENDER_SAMPLES=1
```

## 6. 运行时日志

脚本运行时会：

- 在终端打印总图数和粗略 ETA
- 持续输出动态进度
- 同时把完整日志写到：

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
