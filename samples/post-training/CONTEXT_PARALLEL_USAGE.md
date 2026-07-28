# 可配置 Context Parallel Zigzag / Ulysses 使用指南

本文说明如何在 OmniDreams single-view causal post-training 中使用
FlashAttention-3（FA3）、selective activation checkpoint（SAC）以及
Contiguous、Zigzag、Ulysses context parallel（CP）。

`CP_SIZE=1` 和 `CP_SIZE=4` 只是已经测试过的配置，不是代码中的固定限制。
用户可以按模型形状和可用资源设置：

- `NPROC`：单节点启动的 worker/GPU 数；
- `CP_SIZE`：每个 context-parallel group 的 rank 数；
- `FSDP_SIZE`：每个 FSDP shard group 的 rank 数。

两个通用 launcher 都不要求恰好 4 张 GPU。它们当前是**单节点**入口，
要求 `CP_SIZE` 和 `FSDP_SIZE` 分别整除 `NPROC`，并要求
`NPROC` 不大于 `CUDA_VISIBLE_DEVICES` 中列出的 GPU 数。

> 这是研究和性能分析工作流，不是 release 的最小 8-GPU smoke 路径。
> launcher 会关闭 validation，并把 checkpoint save 替换为 no-op，因此
> 只用于短跑 A/B/profile，不会产出可恢复的训练 checkpoint。标准 E1/E2/E3
> 验证和正式 checkpoint 训练仍应遵循
> [QUICKSTART](./QUICKSTART.md) 和 `torchrun_smoke.sh`。

实现、正确性和 CP=4 的历史性能分析见
[Ulysses / Zigzag 实现与性能报告](./CP4_ULYSSES_ZIGZAG_REPORT.md)。

## 1. 支持边界与已测范围

实现按实际 process-group world size 工作，没有把 CP 固定为 1 或 4。当前
distributed oracle 已在 CP=1、CP=2、CP=3 和 CP=4 上验证；完整训练和性能数字则
来自特定的 4×H20-3e、CP=1/CP=4 实验。尚未实测的 GPU/CP 规模应先运行
第 7 节 oracle 和短跑，再用于长时间作业。

当前训练路径支持：

- single-view causal training；
- BF16/FP16 FA3；
- `patch_temporal=1`；
- `num_interleave=0`；
- `split_cp_in_model=false`；
- FA3 + `predict2_2b_720_aggressive` SAC；
- 用户可配置的 `NPROC`、`CP_SIZE` 和 `FSDP_SIZE`。

当前不支持：

- cross-view/multiview 网络；
- teacher-forcing interleaved sequence；
- 非 SM90 GPU 上的 FA3-NV；
- 推理和 KV-cache 分片不会使用 Zigzag/Ulysses training layout；现有推理
  路径保持不变；
- FlexAttention + Ulysses。

构造期能够识别的 backend/strategy 冲突以及运行期的 shape 整除错误会明确
报错；training layout 不参与 inference/KV-cache 路径，而不是让推理报错。

## 2. 策略、约束和选择

设 CP group size 为 `P=CP_SIZE`，global patched sequence length 为 `S`，
self-attention head 数为 `H_heads`：

| 策略 | 数据布局 | 优点 | 数学约束/代价 |
|---|---|---|---|
| `contiguous` | 每个 rank 持有连续的 `S/P` query | 基线简单，便于 A/B | 要求 `S % P == 0`；causal query 越晚计算越重，rank 负载不均 |
| `zigzag` | 每个 rank 持有首尾对称的两个 micro-chunk | 负载均衡；没有 head divisibility 约束 | 要求 `S % (2P) == 0`；需要 K/V gather、chronological reorder 和额外 DtoD copy |
| `ulysses` | attention 内将 `[B,S/P,H_heads,D]` A2A 成 `[B,S,H_heads/P,D]` | head 维严格均衡 | 要求 `S % P == 0` 且 `H_heads % P == 0`；有 A2A 和额外 NCCL buffer |

三个策略都必须先能把 sequence 均分给 `P` 个 rank，即 `S % P == 0`；
Zigzag 的 `S % (2P) == 0` 是更强的首尾双 chunk 约束。

当前真实网络有 16 个 attention heads，因此 Ulysses 的可选 `CP_SIZE` 必须
同时满足：

```text
16 % CP_SIZE == 0
NPROC % CP_SIZE == 0
```

例如在不考虑其他并行约束时，`CP_SIZE=1/2/4/8/16` 能整除 16；
最终仍要同时满足实际 `NPROC` 和显存/通信条件。Zigzag 不要求 head
可整除，但必须满足：

```text
S % (2 * CP_SIZE) == 0
```

当前 93-frame profile shape 为：

```text
T=24, H=44, W=80
S=84,480
```

选择新的 `CP_SIZE` 时，应根据真实输入、patch 和 head 配置重新验证，不能
仅根据 GPU 数推断可用。

### 历史 CP=4 性能测量

以下是固定 4×H20-3e、`NPROC=4`、`CP_SIZE=4`、`FSDP_SIZE=4`、
16-head、单机互联条件下的测量，不代表其他 CP/GPU 规模：

| FA3 + SAC 策略 | 稳态 mean | 相对 Contiguous |
|---|---:|---:|
| Contiguous | 24.587 s/iter | — |
| Zigzag | 21.103 s/iter | -14.168% |
| Ulysses | 20.852 s/iter | -15.191% |

这些数字来自固定 93-frame 单 clip 重复 200 次的算力 A/B，video 和 HDMap
指向同一 MP4。它们不代表真实 HD-map 数据分布、数据加载吞吐或生成质量。
该次 CP=4 测量中 Ulysses 相对 Zigzag 仅快约 1%，应视为基本打平：

- 优先该配置的最高实测吞吐：选择 Ulysses；
- 优先约束更少、NCCL allocator 外显存更低：选择 Zigzag；
- 做回归或定位负载不均：保留 Contiguous 作为 control。

## 3. 准备环境和数据

先按 [QUICKSTART §1–§3](./QUICKSTART.md#1-prerequisites) 准备 checkpoint
和标准 93-frame 数据：

```bash
export OMNI_CACHE_DIR=/path/to/writable/cache
export UV_CACHE_DIR="$OMNI_CACHE_DIR/uv"
export TMPDIR="$OMNI_CACHE_DIR/tmp"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

bash samples/post-training/setup_env.sh
```

FA3 使用独立的 x86_64 Cosmos Framework 环境，不覆盖 release Torch 2.7
venv。已验证版本为：

| 组件 | 版本 |
|---|---|
| Python | 3.13 |
| PyTorch / CUDA | 2.10.x / 12.8 |
| Transformer Engine | 2.12 |
| FlashAttention-2 | 2.7.4.post1 |
| FlashAttention-3-NV | 1.0.3 |
| NATTEN | 0.21.6 |

完整创建命令见
[README 的 FA3 环境说明](./README.md#optional-configurable-flashattention-3-profiling)。
创建完成后：

```bash
export OMNI_FA3_VENV=/path/to/cosmos-cu128-torch210-venv
```

如果没有显式设置，`fa3_env.sh` 会依次查找：

```text
$OMNI_CACHE_DIR/omnidreams-venvs/fa3-cu128-torch210
$COSMOS_FRAMEWORK_ROOT/.venv-cu128-t210
```

`OMNI_FA3_PYTHON` 默认是 `$OMNI_FA3_VENV/bin/python`。通用 launcher
会依次 source `_env.sh` 和 `fa3_env.sh`，验证上述版本，并覆盖调用方的
`PYTHONPATH`/`LD_LIBRARY_PATH`，避免混入 release Python 或宿主 CUDA
runtime。所有 A/B mode（包括 Flex control）都使用该环境。

默认数据路径是：

```text
post-training/data/{video,hdmap,caption}/
```

使用其他数据时设置 `PROFILE_DATA_ROOT`。该目录必须包含 `video/`、
`hdmap/` 和 `caption/`；single-view 至少要有
`camera_front_wide_120fov`，三个子树中需要同 stem 的 caption/video/HDMap。
视频和 HDMap 必须时间对齐且至少 93 帧。

## 4. 通用 launcher

### 4.1 CP layout A/B

`run_cp_attention_ab.sh` 比较 CP layout，提供五种 mode：

| Mode | Attention backend | CP strategy | Checkpoint policy |
|---|---|---|---|
| `flex-contiguous` | FlexAttention | Contiguous | whole-block |
| `flex-zigzag` | FlexAttention | Zigzag | whole-block |
| `fa3-contiguous` | FA3 | Contiguous | aggressive SAC |
| `fa3-zigzag` | FA3 | Zigzag | aggressive SAC |
| `fa3-ulysses` | FA3 | Ulysses | aggressive SAC |

同一 backend family 内比较 CP strategy 时只改变 layout。Flex 与 FA3 mode
之间同时改变 attention backend 和 checkpoint policy，属于组合收益，不是
纯 backend A/B。

### 4.2 Backend / checkpoint A/B

`run_fa3_attention_ab.sh` 固定使用 Contiguous CP，用于拆分 backend 和 SAC
收益：

| Mode | Attention backend | CP strategy | Checkpoint policy |
|---|---|---|---|
| `flex` | FlexAttention | Contiguous | whole-block |
| `fa3` | FA3 | Contiguous | whole-block |
| `fa3-sac` | FA3 | Contiguous | aggressive SAC |

### 4.3 并行参数和默认值

两个 launcher 都使用相同的并行参数：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | 必须显式设置 | 本次单节点作业可见 GPU 列表，数量不限于 4 |
| `NPROC` | `CUDA_VISIBLE_DEVICES` 条目数 | 单节点 torchrun worker 数 |
| `CP_SIZE` | `NPROC` | context-parallel group size |
| `FSDP_SIZE` | `NPROC` | FSDP shard group size |

约束是：

```text
NPROC <= number of CUDA_VISIBLE_DEVICES entries
NPROC % CP_SIZE == 0
NPROC % FSDP_SIZE == 0
```

当 `NPROC` 小于可见 GPU 数时，torchrun 使用前 `NPROC` 个 local device。
为避免结果归属不清，建议正常作业令两者相等。

### 4.4 启动示例

下面的 CP=1、CP=4 和 CP=8 都只是示例；launcher 不会把这些值写死。

4 GPU、CP=1、FSDP=4，拆分测量 FA3 与 SAC：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC=4 CP_SIZE=1 FSDP_SIZE=4 \
  bash samples/post-training/run_fa3_attention_ab.sh fa3-sac
```

4 GPU、CP=4、FSDP=4，比较 Zigzag/Ulysses：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC=4 CP_SIZE=4 FSDP_SIZE=4 \
  bash samples/post-training/run_cp_attention_ab.sh fa3-zigzag

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC=4 CP_SIZE=4 FSDP_SIZE=4 \
  bash samples/post-training/run_cp_attention_ab.sh fa3-ulysses
```

8 GPU、CP=4、FSDP=8：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC=8 CP_SIZE=4 FSDP_SIZE=8 \
  bash samples/post-training/run_cp_attention_ab.sh fa3-ulysses
```

8 GPU、CP=8、FSDP=8：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC=8 CP_SIZE=8 FSDP_SIZE=8 \
  bash samples/post-training/run_cp_attention_ab.sh fa3-zigzag

# 当前模型有 16 heads，因此 CP=8 也可用于 Ulysses：
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC=8 CP_SIZE=8 FSDP_SIZE=8 \
  bash samples/post-training/run_cp_attention_ab.sh fa3-ulysses
```

换用其他 GPU 数时，显式设置三个并行参数，并重新检查第 2 节的 shape
约束。例如 8 张可见 GPU 中只启动 6 个 worker：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC=6 CP_SIZE=3 FSDP_SIZE=6 \
  bash samples/post-training/run_cp_attention_ab.sh fa3-zigzag
```

该例对当前 `S=84,480` 满足 Zigzag 约束；它不适用于 16-head Ulysses，
因为 `16 % 3 != 0`。

### 4.5 兼容 wrapper

旧文件名仍可用于复现实验，但只作为兼容 wrapper：

- `run_cp4_attention_ab.sh` 只为未设置的 `CP_SIZE` 提供默认值 `4`，
  再转发到 `run_cp_attention_ab.sh`；
- `run_fa3_cp1_ab.sh` 只为未设置的 `CP_SIZE` 提供默认值 `1`，
  再转发到 `run_fa3_attention_ab.sh`。

`NPROC` 仍从可见 GPU 数推断，`FSDP_SIZE` 仍默认等于 `NPROC`；因此四张
可见 GPU 时会复现旧的四卡拓扑，更多 GPU 时不会把 worker 数限制为 4。
调用者可覆盖所有环境变量。新脚本和文档应使用通用 launcher 名称。

### 4.6 其他常用环境变量

| 变量 | 默认值 | 用途 |
|---|---|---|
| `OMNI_FA3_VENV` | 从 cache 或 `COSMOS_FRAMEWORK_ROOT` 查找 | FA3 Python 环境 |
| `COSMOS_FRAMEWORK_ROOT` | 未设置 | 备选 FA3 venv 查找根目录 |
| `OMNI_FA3_PYTHON` | `$OMNI_FA3_VENV/bin/python` | 自定义 FA3 Python 可执行文件 |
| `OMNI_CACHE_DIR` | `$XDG_CACHE_HOME` 或 `$HOME/.cache` | Hugging Face、Triton 和 artifact 根目录 |
| `TRITON_CACHE_BASE` | `$OMNI_CACHE_DIR/triton/$JOB_NAME` | wrapper 会追加 local-rank 后缀 |
| `PROFILE_DATA_ROOT` | `post-training/data` | 93-frame 数据目录 |
| `MAX_ITER` | CP launcher: `10`; FA3 launcher: `8` | 性能 iteration 数 |
| `MASTER_PORT` | CP launcher: `12460`; FA3 launcher: `12450` | torchrun rendezvous port |
| `JOB_NAME` | 自动生成 | 包含实际 NPROC/CP/FSDP/mode 的 job 名 |
| `ARTIFACT_DIR` | 见下文 | 日志/profile 输出 |
| `NSYS` | `0` | 设为 `1` 启用 Nsight Systems |
| `NSYS_BIN` | 从 `PATH` 查找 | 自定义 `nsys` 可执行文件 |
| `PROFILE_FIRST` | CP launcher: `7`; FA3 launcher: `6` | nsys capture 首个 iteration |
| `PROFILE_LAST` | `MAX_ITER` | nsys capture 最后一个 iteration |

默认 artifact 路径随实际 `NPROC` 和 launcher 类型变化：

```text
# run_cp_attention_ab.sh
$OMNI_CACHE_DIR/nsys/omnidreams_df93_${NPROC}gpu/cp_attention/$JOB_NAME

# run_fa3_attention_ab.sh
$OMNI_CACHE_DIR/nsys/omnidreams_df93_${NPROC}gpu/fa3_ab/$JOB_NAME
```

例如运行 20 个 iteration 并添加 Hydra override：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC=8 CP_SIZE=4 FSDP_SIZE=8 \
OMNI_FA3_VENV=/path/to/fa3-venv \
PROFILE_DATA_ROOT=/path/to/df93 \
MAX_ITER=20 \
JOB_NAME=cp4_ulysses_8gpu_20iter \
  bash samples/post-training/run_cp_attention_ab.sh fa3-ulysses \
  optimizer.lr=1e-4
```

launcher 校验：

```text
1 <= PROFILE_FIRST <= PROFILE_LAST <= MAX_ITER
```

尾部参数会原样作为 Hydra override 传入，最后一个同名值生效。不要用尾部
override 改写 backend、CP strategy、SAC、data root、max-iter 或
`trainer.grad_accum_iter` 等 control 维度：

- `metadata.txt` 记录 launcher 选择和环境变量，不会解析尾部同名 Hydra 值；
- profiling entry 假设 `trainer.grad_accum_iter=1`，更大的 accumulation
  会让 iteration NVTX range 嵌套/错配。

## 5. 关键配置

launcher 根据环境变量设置：

```text
model.config.fsdp_shard_size=${FSDP_SIZE}
model_parallel.context_parallel_size=${CP_SIZE}
model.config.split_cp_in_model=false

model.config.net.training_attention_backend=flex|flash_attn_3
model.config.net.training_context_parallel_strategy=contiguous|zigzag|ulysses
model.config.net.sac_config.mode=block_wise|predict2_2b_720_aggressive
model.config.net.sac_config.every_n_blocks=1
```

在其他 launcher 中接入时也必须保持 `split_cp_in_model=false`。原始 video
和 timestep tensor 先 broadcast 完整序列，再由 causal network 按所选
layout 分片；如果外层和网络内同时分片，attention 会收到错误的 sequence。

要把这些配置用于正式训练，需要在匹配的 FA3 环境中接入正常 training
entry，并保留 validation/checkpointer。不要复用 `fa3_profile_entry.py`，
也不要把当前 profiling launcher 当作正式训练 launcher。

### 5.1 实现入口

| 文件 | 作用 |
|---|---|
| `imaginaire/utils/context_parallel.py` | Zigzag split、inverse gather 和 permutation cache |
| `modules/block_causal_flash_attention.py` | Contiguous/Zigzag query-sharded FA3 与 Ulysses FA3 dispatch |
| `modules/ulysses_attention.py` | packed QKV sequence-to-head A2A 及 autograd 逆变换 |
| `modules/flex_attention.py` | Zigzag physical/logical mask index 映射 |
| `networks/causal_cosmos.py` | single-view strategy 配置、分片、attention 和输出恢复 |
| `networks/causal_cosmos_hdmap.py` | HDMap 网络的相同训练路径 |
| `models/joint_causal_cosmos_model.py` | `split_cp_in_model` 防二次分片校验 |
| `networks/causal_crossview_cosmos.py` | 明确拒绝尚未支持的 multiview 组合 |

## 6. Nsight Systems profiling

例如在 8 GPU、CP=4 上捕获 clean iteration：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC=8 CP_SIZE=4 FSDP_SIZE=8 \
NSYS=1 PROFILE_FIRST=7 PROFILE_LAST=8 MAX_ITER=8 \
  bash samples/post-training/run_cp_attention_ab.sh fa3-ulysses
```

CP launcher 默认从 iteration 7 开始，是为了避开 iteration 5 的一次性
`tokenizer.encode` lazy compile。不要用包含 iteration 5 的窗口计算稳态
吞吐。

每个 local rank 启动一个独立 nsys process，使用 CUDA profiler API 捕获
`PROFILE_FIRST..PROFILE_LAST`（包含首尾）。capture 开始前以及每个被捕获
iteration 末尾都有诊断同步，因此 profile wall time 会受扰动。trace 只包含
CUDA/NVTX，不启用 CPU sampling、context-switch sampling 或 backtrace。

每个 artifact 目录包含：

- `train.log`；
- `metadata.txt`：实际 NPROC/CP/FSDP、模式、版本、GPU 和源码 hash；
- `git-status.txt` 和 `tracked-changes.patch`；
- `nvml.csv`：200 ms 显存/utilization 采样；
- `timing_rank0.nsys-rep` 到 `timing_rank$((NPROC-1)).nsys-rep`
  （`NSYS=1` 时）。

`nvml.csv` 由宿主 `nvidia-smi` 生成，并通过 `-i` 限制为
`CUDA_VISIBLE_DEVICES` 中列出的卡。若 `NPROC` 小于可见 GPU 数，文件仍会
包含未被 worker 使用的可见卡，因此建议二者数量保持一致。launcher 不会
自动把 `.nsys-rep` 导出为 SQLite 或生成 analysis Markdown。

端到端性能应使用非 nsys 的稳态 iteration。nsys 用于解释 kernel、通信和
rank balance，不应替代无 profiler 的 headline。

## 7. 正确性验证

### 7.1 Samples tests

```bash
export OMNI_CACHE_DIR=/path/to/cache
export OMNI_FA3_VENV=/path/to/cosmos-cu128-torch210-venv
source samples/post-training/_env.sh
source samples/post-training/fa3_env.sh

"$OMNI_FA3_PYTHON" -m pytest -q samples/post-training/tests
```

如果 canonical `post-training/.venv` 没有创建，三个 release CUDA-stack
inventory tests 会 skip；FA3 Hopper forward/gradient tests 不应 skip。

### 7.2 N-rank distributed oracle

oracle 使用 CP=1 full-sequence FA3 作为 reference，对 Contiguous、Zigzag
和 Ulysses 的 full output 及 local `dQ/dK/dV` 做比较。进程数就是被测
`CP_SIZE`；脚本没有固定四卡检查。它会按 CP size 构造可整除的 synthetic
sequence/head shape，因此 CP=3 的 Ulysses oracle 通过不表示真实 16-head
模型可使用 CP=3；完整训练仍需满足第 2 节的模型约束。

```bash
export OMNI_CACHE_DIR=/path/to/cache
export OMNI_FA3_VENV=/path/to/cosmos-cu128-torch210-venv
source samples/post-training/_env.sh
source samples/post-training/fa3_env.sh

cd post-training
CUDA_VISIBLE_DEVICES=<comma-separated-GPU-list> \
"$OMNI_FA3_PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=<CP_SIZE> \
  ../samples/post-training/tests/torchrun_cp_attention_correctness.py
```

已经执行过的 CP=1/2/3/4 回归示例：

```bash
CUDA_VISIBLE_DEVICES=0 \
"$OMNI_FA3_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=1 \
  ../samples/post-training/tests/torchrun_cp_attention_correctness.py

CUDA_VISIBLE_DEVICES=0,1 \
"$OMNI_FA3_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 \
  ../samples/post-training/tests/torchrun_cp_attention_correctness.py

CUDA_VISIBLE_DEVICES=0,1,2 \
"$OMNI_FA3_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=3 \
  ../samples/post-training/tests/torchrun_cp_attention_correctness.py

CUDA_VISIBLE_DEVICES=0,1,2,3 \
"$OMNI_FA3_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 \
  ../samples/post-training/tests/torchrun_cp_attention_correctness.py
```

每次预期输出中的 `P` 等于本次 `--nproc-per-node`：

```text
PASS: contiguous CP=P forward and gradients
PASS: zigzag CP=P forward and gradients
PASS: ulysses CP=P forward and gradients
```

在尝试 CP=8、CP=16 或其他新规模前，先用对应 rank 数运行 oracle；oracle
通过只说明 attention layout 数值正确，不替代完整模型短跑、显存和性能验证。

## 8. 公平 A/B 方法

建议至少：

1. 固定 checkpoint、数据、GPU、`NPROC`、`CP_SIZE`、`FSDP_SIZE` 和所有
   Hydra overrides；
2. 第一轮运行 `contiguous → zigzag → ulysses`；
3. 第二轮反向运行 `ulysses → zigzag → contiguous`；
4. 排除 iteration 1 cold start 和 iteration 5 lazy compile；
5. 优先统计 compile 后的 iteration 6+；
6. 每个策略至少收集 20 个稳态 iteration，并做三个 job-level repeats；
7. 同时记录 iteration time、loss、PyTorch peak 和 NVML peak。

launcher 默认按 `JOB_NAME` 创建独立 Triton cache，因此每个新 job 都可能
经历 cold compile。若要共享 warm cache，应显式设置相同
`TRITON_CACHE_BASE`，仍由 wrapper 为每个 local rank 追加独立后缀。
`JOB_NAME`/`ARTIFACT_DIR` 应保持唯一，避免覆盖日志或遗留旧 profile。

不要用以下指标直接代替端到端收益：

- 所有 GPU kernel duration 的简单求和；
- NCCL kernel lifetime（可能包含等待并与 compute overlap）；
- launch-to-start queue（GPU 满载时主要表示 enqueue depth）；
- 单个 rank 的时间。

## 9. 常见错误

### `CP_SIZE ... must divide NPROC`

`CP_SIZE` 不是 GPU 总数的别名，但必须整除本次 `NPROC`。调整二者，使每个
worker 都能加入完整 CP group。

### `FSDP_SIZE ... must divide NPROC`

调整 `FSDP_SIZE` 或 `NPROC`，使每个 worker 都能加入完整 FSDP group。

### `NPROC ... exceeds ... CUDA_VISIBLE_DEVICES`

增大可见 GPU 列表或减小 `NPROC`。当前 launcher 是单节点入口，不会跨节点
补足 worker。

### `requires model.config.split_cp_in_model=false`

FA3 CP 或非 Contiguous 策略需要 network 持有完整 sequence 后自行分片：

```text
model.config.split_cp_in_model=false
```

### `num_heads ... divisible by CP size`

Ulysses 的 head shard 不能整除。当前模型为 16 heads；选择能整除 16 的
`CP_SIZE`，或改用 Zigzag。

### Sequence divisibility error

Contiguous/Ulysses 的 global patched sequence 必须能被 `CP_SIZE` 整除；
Zigzag 必须能被 `2 * CP_SIZE` 整除。调整输入 shape、patch 配置或 CP size。

### `FlashAttention-3 ... unavailable`

当前 Python 环境缺少 `flash-attn-3-nv`，或没有通过 `fa3_env.sh` 进入
匹配 Torch/CUDA ABI 的环境。不要把 FA3 wheel 单独安装到 release
Torch 2.7 venv。

### `requires an NVIDIA Hopper GPU`

FA3-NV 路径要求 SM90。非 Hopper GPU 使用标准 release/Flex 路径。

### Cross-view constructor rejects the strategy

当前实现只接入 single-view causal self-attention。multiview 需要单独设计
view/sequence layout 和 cross-view communication，不能直接启用。

### Dataset directory is missing

运行 `samples/post-training/setup_env.sh`，或设置包含 `video/`、`hdmap/`、
`caption/` 的 `PROFILE_DATA_ROOT`。

### `NSYS=1 requires nsys`

把 `nsys` 加入 `PATH`，或设置可执行的 `NSYS_BIN=/path/to/nsys`。只有字符串
`NSYS=1` 会启用 capture，`NSYS=true` 不会。

### `PROFILE_FIRST <= PROFILE_LAST <= MAX_ITER`

三个值必须是正整数并满足该顺序。若把 `MAX_ITER` 调到小于默认
`PROFILE_FIRST`，也要同时降低 `PROFILE_FIRST`。

### FA3 environment mismatch

`fa3_env.sh` 检测到 Python/Torch/CUDA/TE/FA2/FA3/NATTEN 版本与已验证组合
不一致。使用 Cosmos Framework pinned environment，不要绕过版本校验。

### `patch_temporal=1` / `num_interleave=0`

FA3 block-causal training 当前只支持 `patch_temporal=1` 且
`num_interleave=0`。teacher-forcing interleaved layout 继续使用受支持的
Flex/release 路径。

### Unsupported FA3 dtype

FA3 training 输入必须是 BF16 或 FP16。检查 autocast、权重和 activation
dtype 是否被尾部 override 改写。

### Rendezvous port is already in use

为并发 job 设置不同的 `MASTER_PORT`，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC=8 CP_SIZE=4 FSDP_SIZE=8 MASTER_PORT=12461 \
  bash samples/post-training/run_cp_attention_ab.sh fa3-zigzag
```

## 10. 默认行为

release 配置默认仍是：

```text
training_attention_backend=flex
training_context_parallel_strategy=contiguous
```

因此合入代码不会无意改变现有实验语义。只有显式选择专用 launcher 或传入
对应 Hydra overrides 时，才启用 FA3、Zigzag 或 Ulysses。
