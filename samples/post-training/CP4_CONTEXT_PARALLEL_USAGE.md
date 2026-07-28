# CP=4 Zigzag / Ulysses 使用指南

本文说明如何在 OmniDreams single-view causal post-training 中使用
FlashAttention-3（FA3）、selective activation checkpoint（SAC）以及
CP=4 Zigzag/Ulysses context parallel。

> 这是经过 4 张 H20-3e 验证的研究和性能分析工作流，不是 release
> 支持的最小 8-GPU smoke 路径。`run_cp4_attention_ab.sh` 会关闭
> validation 并把 checkpoint save 替换为 no-op，因此只用于短跑 A/B/profile，
> 不会产出可恢复的训练 checkpoint。标准 E1/E2/E3 验证和正式 checkpoint
> 训练仍应遵循
> [QUICKSTART](./QUICKSTART.md) 和 `torchrun_smoke.sh`。

实现、正确性和性能分析见
[CP=4 Ulysses / Zigzag 实现与性能报告](./CP4_ULYSSES_ZIGZAG_REPORT.md)。

## 1. 已验证范围

当前已验证组合包括：

- single-view causal training；
- 4 张 Hopper/SM90 GPU；性能和完整训练实测使用 H20-3e；
- CP=4、FSDP shard size=4；
- BF16/FP16 FA3；
- `patch_temporal=1`；
- `num_interleave=0`；
- `split_cp_in_model=false`；
- FA3 + `predict2_2b_720_aggressive` SAC。

当前不支持：

- cross-view/multiview 网络；
- teacher-forcing interleaved sequence；
- 非 SM90 GPU；
- 将 Zigzag/Ulysses training layout 用于推理或 KV-cache 分片；现有推理
  路径保持不变；
- FlexAttention + Ulysses。

不支持的组合会 fail fast，不会静默退化到另一种 CP 策略。

## 2. 策略选择

| 策略 | 数据布局 | 优点 | 约束/代价 |
|---|---|---|---|
| `contiguous` | 每个 rank 持有连续的 `S/CP` query | 基线简单，便于 A/B | causal query 越晚计算越重，rank 负载不均 |
| `zigzag` | 每个 rank 持有首尾对称的两个 micro-chunk | 负载均衡；没有 head divisibility 约束 | 需要 K/V gather、chronological reorder 和额外 DtoD copy |
| `ulysses` | attention 内将 `[B,S/CP,H,D]` A2A 成 `[B,S,H/CP,D]` | head 维严格均衡；当前 CP=4 吞吐略高 | 要求 `num_heads % cp_size == 0`；有 A2A 和额外 NCCL buffer |

当前 4×H20-3e、16-head、单机互联实验中：

| FA3 + SAC 策略 | 稳态 mean | 相对 Contiguous |
|---|---:|---:|
| Contiguous | 24.587 s/iter | — |
| Zigzag | 21.103 s/iter | -14.168% |
| Ulysses | 20.852 s/iter | -15.191% |

这些数字来自固定 93-frame 单 clip 重复 200 次的算力 A/B，video 和 HDMap
指向同一 MP4。它们不代表真实 HD-map 数据分布、数据加载吞吐或生成质量。

Ulysses 相对 Zigzag 仅快约 1%，应视为基本打平：

- 优先当前配置最高吞吐：选择 Ulysses；
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

CP=4 FA3 使用独立的 x86_64 Cosmos Framework 环境，不覆盖 release
Torch 2.7 venv。已验证版本为：

| 组件 | 版本 |
|---|---|
| Python | 3.13 |
| PyTorch / CUDA | 2.10.x / 12.8 |
| Transformer Engine | 2.12 |
| FlashAttention-2 | 2.7.4.post1 |
| FlashAttention-3-NV | 1.0.3 |
| NATTEN | 0.21.6 |

完整创建命令见
[README 的 CP=1 FA3 环境说明](./README.md#optional-cp1-flashattention-3-profiling)。
创建完成后：

```bash
export OMNI_FA3_VENV=/path/to/cosmos-cu128-torch210-venv
```

如果没有显式设置，`fa3_env.sh` 会依次查找：

```text
$OMNI_CACHE_DIR/omnidreams-venvs/fa3-cu128-torch210
$COSMOS_FRAMEWORK_ROOT/.venv-cu128-t210
```

`OMNI_FA3_PYTHON` 默认是 `$OMNI_FA3_VENV/bin/python`，通常不需要单独设置。
统一 launcher 会依次 source `_env.sh` 和 `fa3_env.sh`，验证上述版本，并覆盖
调用方的 `PYTHONPATH`/`LD_LIBRARY_PATH`，避免混入 release Python 或宿主
CUDA runtime。五种模式（包括 Flex control）都使用该环境，确保 A/B 软件栈
一致。

默认数据路径是：

```text
post-training/data/{video,hdmap,caption}/
```

使用其他数据时设置 `PROFILE_DATA_ROOT`。该目录必须包含 `video/`、
`hdmap/` 和 `caption/`；single-view 至少要有
`camera_front_wide_120fov`，三个子树中需要同 stem 的 caption/video/HDMap。
视频和 HDMap 必须时间对齐且至少 93 帧。launcher 只预检三个顶层目录，
实际文件匹配由 dataloader 验证。

## 4. 启动性能实验

launcher 的五种 mode：

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

所有 mode 都会：

- 使用 `fa3_profile_entry.py`；
- 设置 `trainer.run_validation=false`；
- 设置 `dataloader_train.val_holdout_frac=0.0` 和 `repeat_factor=200`；
- 禁用实际 checkpoint save。

因此下列命令用于性能/正确性短跑，不用于长期训练或生成 checkpoint。

### 4.1 推荐模式

从仓库根目录执行：

```bash
# Contiguous control
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash samples/post-training/run_cp4_attention_ab.sh fa3-contiguous

# Zigzag
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash samples/post-training/run_cp4_attention_ab.sh fa3-zigzag

# Ulysses
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash samples/post-training/run_cp4_attention_ab.sh fa3-ulysses
```

`CUDA_VISIBLE_DEVICES` 必须恰好列出四张空闲 GPU。物理 GPU 编号不需要从
0 开始，例如：

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 \
  bash samples/post-training/run_cp4_attention_ab.sh fa3-ulysses
```

### 4.2 FlexAttention control

launcher 也提供两个诊断 control：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash samples/post-training/run_cp4_attention_ab.sh flex-contiguous

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash samples/post-training/run_cp4_attention_ab.sh flex-zigzag
```

Flex 模式使用 whole-block checkpoint。Ulysses 没有 Flex 模式。

### 4.3 常用环境变量

| 变量 | 默认值 | 用途 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | 必须显式设置 | 恰好四张训练 GPU |
| `OMNI_FA3_VENV` | 从 cache 或 `COSMOS_FRAMEWORK_ROOT` 查找 | FA3 Python 环境 |
| `COSMOS_FRAMEWORK_ROOT` | 未设置 | 备选 FA3 venv 查找根目录 |
| `OMNI_FA3_PYTHON` | `$OMNI_FA3_VENV/bin/python` | 自定义 FA3 Python 可执行文件 |
| `OMNI_CACHE_DIR` | `$XDG_CACHE_HOME` 或 `$HOME/.cache` | Hugging Face、Triton 和 profile artifact 根目录 |
| `TRITON_CACHE_BASE` | `$OMNI_CACHE_DIR/triton/$JOB_NAME` | wrapper 会追加 local-rank 后缀 |
| `PROFILE_DATA_ROOT` | `post-training/data` | 93-frame 数据目录 |
| `MAX_ITER` | `10` | 性能 iteration 数 |
| `MASTER_PORT` | `12460` | torchrun rendezvous port |
| `JOB_NAME` | 自动生成 | job 和 artifact 名称 |
| `ARTIFACT_DIR` | `$OMNI_CACHE_DIR/nsys/omnidreams_df93_4gpu/cp4_attention/$JOB_NAME` | 日志/profile 输出 |
| `NSYS` | `0` | 设为 `1` 启用 Nsight Systems |
| `NSYS_BIN` | 从 `PATH` 查找 | 自定义 `nsys` 可执行文件 |
| `PROFILE_FIRST` | `7` | nsys capture 首个 iteration |
| `PROFILE_LAST` | `MAX_ITER` | nsys capture 最后一个 iteration |

例如运行 20 个性能 iteration：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMNI_FA3_VENV=/path/to/fa3-venv \
PROFILE_DATA_ROOT=/path/to/df93 \
MAX_ITER=20 \
JOB_NAME=cp4_ulysses_20iter \
  bash samples/post-training/run_cp4_attention_ab.sh fa3-ulysses
```

launcher 校验：

```text
1 <= PROFILE_FIRST <= PROFILE_LAST <= MAX_ITER
```

尾部参数会原样作为 Hydra override 传入，最后一个同名值生效。例如：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MAX_ITER=20 \
  bash samples/post-training/run_cp4_attention_ab.sh fa3-zigzag \
  optimizer.lr=1e-4
```

不要用尾部 override 改写 backend、CP strategy、SAC、data root、max-iter
或 `trainer.grad_accum_iter` 等 control 维度：

- `metadata.txt` 记录 launcher 选择和环境变量，不会解析尾部同名 Hydra 值；
- profiling entry 假设 `trainer.grad_accum_iter=1`，更大的 accumulation
  会让 iteration NVTX range 嵌套/错配。

## 5. 关键配置

FA3 modes 设置以下并行和训练配置：

```text
model.config.fsdp_shard_size=4
model_parallel.context_parallel_size=4
model.config.split_cp_in_model=false

model.config.net.training_attention_backend=flash_attn_3
model.config.net.training_context_parallel_strategy=contiguous|zigzag|ulysses
model.config.net.sac_config.mode=predict2_2b_720_aggressive
model.config.net.sac_config.every_n_blocks=1
```

在其他 launcher 中接入时也必须保持 `split_cp_in_model=false`。原始 video
和 timestep tensor 先 broadcast 完整序列，再由 causal network 按所选
layout 分片；如果外层和网络内同时分片，attention 会收到错误的 sequence。

要把这些配置用于正式训练，需要在匹配的 FA3 环境中接入正常 training
entry，并保留 validation/checkpointer。不要复用 `fa3_profile_entry.py`，
也不要把当前 profiling launcher 当作正式训练 launcher。

Zigzag 还要求 global patched sequence 可以均分为 `2 * cp_size` 个
micro-chunk。当前验证/profile shape：

```text
T=24, H=44, W=80
sequence=84,480
2 * CP=8
```

满足该约束。

Ulysses 要求：

```text
num_heads % cp_size == 0
```

当前 `16 % 4 == 0`。如果换模型宽度或 CP size，应先检查 head 数。

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

launcher 使用 CUDA profiler API，只捕获指定的 clean iteration：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NSYS=1 \
PROFILE_FIRST=7 \
PROFILE_LAST=8 \
MAX_ITER=8 \
  bash samples/post-training/run_cp4_attention_ab.sh fa3-ulysses
```

默认从 iteration 7 开始，是为了避开 iteration 5 的一次性
`tokenizer.encode` lazy compile。不要用包含 iteration 5 的窗口计算稳态
吞吐。

每个 local rank 启动一个独立 nsys process，使用 CUDA profiler API 捕获
`PROFILE_FIRST..PROFILE_LAST`（包含首尾）。capture 开始前以及每个被捕获
iteration 末尾都有诊断同步，因此 profile wall time 会受扰动。trace 只包含
CUDA/NVTX，不启用 CPU sampling、context-switch sampling 或 backtrace。

每个 artifact 目录包含：

- `train.log`；
- `metadata.txt`：模式、版本、GPU、数据 symlink/hash、源码 hash；
- `git-status.txt` 和 `tracked-changes.patch`；
- `nvml.csv`：200 ms 显存/utilization 采样；
- `timing_rank0..3.nsys-rep`（`NSYS=1` 时）。

`nvml.csv` 由宿主 `nvidia-smi` 生成，会列出整台机器的 GPU，而不仅是
`CUDA_VISIBLE_DEVICES` 中的四张卡；分析时按物理 GPU index 过滤。launcher
不会自动把 `.nsys-rep` 导出为 SQLite 或生成 analysis Markdown。

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

### 7.2 四卡 distributed oracle

该测试用 CP=1 full-sequence FA3 作为 reference，对 Contiguous、Zigzag 和
Ulysses 的 full output 及 local `dQ/dK/dV` 做比较：

```bash
export OMNI_CACHE_DIR=/path/to/cache
export OMNI_FA3_VENV=/path/to/cosmos-cu128-torch210-venv
source samples/post-training/_env.sh
source samples/post-training/fa3_env.sh

cd post-training
CUDA_VISIBLE_DEVICES=0,1,2,3 \
"$OMNI_FA3_PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  ../samples/post-training/tests/torchrun_cp_attention_correctness.py
```

预期：

```text
PASS: contiguous CP=4 forward and gradients
PASS: zigzag CP=4 forward and gradients
PASS: ulysses CP=4 forward and gradients
```

## 8. 公平 A/B 方法

建议至少：

1. 固定 checkpoint、数据、GPU 和所有 Hydra overrides；
2. 第一轮运行 `contiguous → zigzag → ulysses`；
3. 第二轮反向运行 `ulysses → zigzag → contiguous`；
4. 排除 iteration 1 cold start 和 iteration 5 lazy compile；
5. 优先统计 compile 后的 iteration 6+；
6. 每个策略至少收集 20 个稳态 iteration，并做三个 job-level repeats；
7. 同时记录 iteration time、loss、PyTorch peak 和 NVML peak。

launcher 默认按 `JOB_NAME` 创建独立 Triton cache，因此每个新 job 都可能
经历 cold compile。若要共享 warm cache，应显式设置相同
`TRITON_CACHE_BASE`，仍由 wrapper 为每个 local rank 追加独立后缀。
`JOB_NAME`/`ARTIFACT_DIR` 应保持唯一，避免覆盖核心日志或遗留旧 profile。

不要用以下指标直接代替端到端收益：

- 所有 GPU kernel duration 的简单求和；
- NCCL kernel lifetime（可能包含等待并与 compute overlap）；
- launch-to-start queue（GPU 满载时主要表示 enqueue depth）；
- 单个 rank 的时间。

## 9. 常见错误

### `requires model.config.split_cp_in_model=false`

FA3 CP 或非 Contiguous 策略需要 network 持有完整 sequence 后自行分片：

```text
model.config.split_cp_in_model=false
```

### `num_heads ... divisible by CP size`

Ulysses 的 head shard 不能整除。选择 Zigzag，或调整模型 head 数/CP size。

### `FlashAttention-3 ... unavailable`

当前 Python 环境缺少 `flash-attn-3-nv`，或没有通过 `fa3_env.sh` 进入
匹配 Torch/CUDA ABI 的环境。不要把 FA3 wheel 单独安装到 release
Torch 2.7 venv。

### `requires an NVIDIA Hopper GPU`

FA3-NV 路径要求 SM90。非 Hopper GPU 使用标准 release/Flex 路径。

### Zigzag sequence divisibility error

global patched sequence 必须能被 `2 * cp_size` 整除。调整输入 shape、patch
配置或 CP size。

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
`PROFILE_FIRST=7`，也要同时降低 `PROFILE_FIRST`。

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
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=12461 \
  bash samples/post-training/run_cp4_attention_ab.sh fa3-zigzag
```

## 10. 默认行为

release 配置默认仍是：

```text
training_attention_backend=flex
training_context_parallel_strategy=contiguous
```

因此合入代码不会无意改变现有实验语义。只有显式选择专用 launcher 或传入
对应 Hydra overrides 时，才启用 FA3、Zigzag 或 Ulysses。
