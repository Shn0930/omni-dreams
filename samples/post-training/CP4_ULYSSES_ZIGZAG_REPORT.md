# CP=4 Ulysses / Zigzag Context Parallel 实现与性能报告

日期：2026-07-28

## 结论

本分支在单视角 causal training 路径中实现了两种 CP=4 负载均衡策略：

- **Zigzag CP**：每个 rank 持有首尾对称的两个 sequence micro-chunk，继续采用 local-Q、full-K/V 的 query-sharded attention。
- **Ulysses CP**：在 self-attention 内通过 all-to-all 将 sequence shard 临时转换为 head shard，计算完整 causal sequence 后再转换回来。

两种实现均通过 CP=4 forward、`dQ/dK/dV` 数值 oracle 和完整 93 帧训练。两轮方向相反的 A/B 中，排除一次性 tokenizer 编译 step 后：

| 模式 | 12-sample mean | 相对 Contiguous 降时 | 吞吐 |
|---|---:|---:|---:|
| FA3 Contiguous | 25.823 s | — | 1.0000x |
| FA3 Zigzag | 22.311 s | 13.602% | 1.1574x |
| FA3 Ulysses | 22.088 s | 14.467% | 1.1691x |

Ulysses 两轮都最快，但相对 Zigzag 的合并优势只有约 **1.0%**。对当前 CP=4、16-head、单机 NVLink 配置，二者都可用；若只追求当前配置的最高吞吐可选 Ulysses，若更看重实现简单、无 `num_heads % cp_size` 约束和更低的 NCCL allocator 外显存，可选 Zigzag。默认配置仍保留 `contiguous`，避免无意改变已有实验语义。

## 实验配置

| 项目 | 配置 |
|---|---|
| GPU | 4 × NVIDIA H20-3e，物理 GPU 1/2/3/4 |
| 并行 | CP=4，FSDP shard size=4，DP=1 |
| 软件 | Python 3.13.13，PyTorch 2.10.0+cu128，TE 2.12，FA3-NV 1.0.3 |
| 输入 | 单视角 93 帧，704×1280，batch size 1 |
| Latent/token | `T=24, H=44, W=80`，总 sequence 84,480 |
| Causal block | 2 latent frames/block，共 12 blocks，每 block 7,040 tokens |
| Attention | 16 heads，head dim 128，FA3 self-attention |
| Checkpoint | SAC `predict2_2b_720_aggressive`，每个 transformer block |
| DF | `noise_scheme=diffusion_forcing`，`state_t=24`，`num_frame_per_block=2` |
| 其他 | `split_cp_in_model=false`，validation/checkpoint save 关闭 |

性能数据集是一条固定 clip 重复 200 次；video 与 hdmap 指向同一 MP4。它适合固定 shape 的算力 A/B，不代表真实 HD-map 数据分布、数据加载吞吐或生成质量。

## 为什么 Contiguous CP 不均衡

CP=4 连续均分后，每个 rank 都持有 21,120 个 query tokens，即 3 个 temporal blocks。但 causal mask 让越靠后的 query 看到越长的 K/V prefix：

| Rank | Query blocks | 合法 block-pair 工作量 |
|---|---|---:|
| 0 | B0–B2 | 1+2+3 = 6 |
| 1 | B3–B5 | 4+5+6 = 15 |
| 2 | B6–B8 | 7+8+9 = 24 |
| 3 | B9–B11 | 10+11+12 = 33 |

所以 sequence 长度虽相同，实际 attention score 计算量为 `6:15:24:33`。低 rank 随后在 FSDP/NCCL collective 中等待 rank 3，表现为很长的 NCCL kernel lifetime。

## 实现

### Zigzag

完整 sequence 被切成 `2 × CP = 8` 个等长 micro-chunks。CP rank `r` 持有：

```text
(r, 2 * CP - r - 1)

rank 0: C0 + C7
rank 1: C1 + C6
rank 2: C2 + C5
rank 3: C3 + C4
```

当前 shape 中每个 micro-chunk 为 10,560 tokens，即 1.5 个 temporal blocks。四个 rank 的理论 causal block-pair 工作量都严格等于 19.5。

实现包含：

- 对 hidden state、timestep embedding、RoPE、AdaLN-LoRA 和 extra positional embedding 使用相同 zigzag permutation。
- Differentiable K/V all-gather 后恢复 chronological order。
- 本地 Q micro-chunk 在 temporal block 边界拆分，每个 segment 只调用其合法 K/V prefix 的 FA3。
- transformer blocks 后 inverse-permute 并 gather，确保 FinalLayer 仍接收原始时间顺序。
- FlexAttention 路径也支持 physical-to-logical mask index 映射；headline A/B 使用 FA3。
- `(device, cp_size)` permutation index 缓存，避免每层从 Python list 重建 CUDA index。

### Ulysses

attention 外部仍保留 local-sequence/all-head layout：

```text
[B, 21,120, 16, 128]
    -- packed QKV all-to-all -->
[B, 84,480, 4, 128]
    -- full-sequence block-causal FA3 -->
[B, 84,480, 4, 128]
    -- inverse all-to-all -->
[B, 21,120, 16, 128]
```

每个 rank 对完整 12-block sequence 计算 4 个 heads，因此工作量严格相同。Q/K/V 被 pack 后只执行一次 sequence-to-head A2A，输出再执行一次 head-to-sequence A2A；custom autograd 在 backward 中执行对应逆变换。

### 配置和作用域

新增训练配置：

```text
training_attention_backend = flex | flash_attn_3
training_context_parallel_strategy = contiguous | zigzag | ulysses
```

约束：

- Ulysses 当前要求 `training_attention_backend=flash_attn_3`。
- Ulysses 要求 `num_heads % cp_size == 0`。
- FA3 当前只支持 `num_interleave=0`、`patch_temporal=1`。
- FA3 CP、Zigzag 和 Ulysses 都要求 `split_cp_in_model=false`，防止外层和网络内二次切分。
- 当前仅接入 single-view causal training；cross-view/multiview 构造时会明确拒绝。
- 推理和 KV-cache 路径未改变。

## 正确性验证

### 单元与 GPU 测试

- Zigzag split + inverse gather 恒等。
- Zigzag inverse permutation 可微。
- 缓存后的 permutation index 被复用。
- 带 per-rank padding 的 Flex mask physical/logical index 映射。
- FA3 对 dense block-causal reference 的 BF16/FP16 forward 和 gradients。
- Samples test suite：**45 passed，3 skipped**。

### 四卡 distributed oracle

固定 full Q/K/V 和 output gradient，对比 CP=1 FA3 oracle：

```text
PASS: contiguous CP=4 forward and gradients
PASS: zigzag CP=4 forward and gradients
PASS: ulysses CP=4 forward and gradients
```

验证项为 full output 和 local `dQ/dK/dV`；容差分别为 `3e-2` 和 `6e-2`。

### 完整训练

两轮 8-iteration A/B 的日志 loss 序列逐 step 一致（以日志打印的四位小数为准）：

```text
0.0544, 0.0603, 0.0690, 0.0604, 0.0651, 0.0840, 0.0713, 0.0678
```

Flex Zigzag 也额外完成 2-iteration full-training smoke；Iteration 2 为 29.58 s，loss 0.0602。它明显慢于 FA3 Zigzag，并使用更多显存，因此不作为推荐性能路径。

## 端到端 A/B

执行了两个 job-level repeats：

1. Contiguous → Zigzag → Ulysses。
2. Ulysses → Zigzag → Contiguous。

每轮统计 iterations `{2,3,4,6,7,8}`。Iteration 5 被排除，因为 release config 在该 step 将 `tokenizer.encode` 切换为 `torch.compile`，首次调用会产生约 6.4 秒的一次性 lazy compile。

### 每轮结果

| 模式 | Run 1 mean | Run 2 mean | 两轮变化 |
|---|---:|---:|---:|
| Contiguous | 25.818 s | 25.828 s | +0.039% |
| Zigzag | 22.415 s | 22.207 s | -0.929% |
| Ulysses | 22.063 s | 22.112 s | +0.219% |

Contiguous 移到最后执行后只变化 0.039%，Ulysses 移到最先后只变化 0.219%，说明主收益不是固定执行顺序造成的。

### 更接近长训练的已编译阶段

合并两轮、只看 tokenizer compile 完成后的 iterations 6–8：

| 模式 | Mean | 相对 Contiguous |
|---|---:|---:|
| Contiguous | 24.587 s | — |
| Zigzag | 21.103 s | -14.168% |
| Ulysses | 20.852 s | -15.191% |

## 显存

第二轮 A/B 的峰值：

| 模式 | PyTorch allocated | PyTorch reserved | NVML sampled peak |
|---|---:|---:|---:|
| Contiguous | 22.452 GiB | 27.211 GiB | 29.429 GiB |
| Zigzag | 22.452 GiB | 27.211 GiB | 29.429 GiB |
| Ulysses | 21.969 GiB | 26.918 GiB | 30.294 GiB |

Ulysses 的 PyTorch allocated 少 0.483 GiB，但 NVML 总显存比另外两种高约 0.865 GiB，符合额外 NCCL/A2A allocator 外 buffer 的表现。

Flex Zigzag smoke 的 PyTorch allocated/reserved 为 33.339/37.830 GiB，进一步说明 FA3+SAC 是当前更合适的组合。

## 修复后 nsys 结果

nsys 对三种模式的四个 rank 都抓取了 clean iterations 7–8。自动导出、标准分析和自定义 SQL 覆盖 GPU kernels、CUDA API、NVTX、memory copy、NCCL 与 launch queue。

### Model-only 和 GPU span

| 模式 | `net_forward.start → optimizer.end` | Net GPU span/iter | Backward GPU span/iter |
|---|---:|---:|---:|
| Contiguous | 13.667 s/iter | 3.907 s | 9.728 s |
| Zigzag | 10.039 s/iter | 2.817 s | 7.189 s |
| Ulysses | 9.956 s/iter | 2.765 s | 7.157 s |

相对 Contiguous，model-only 窗口：

- Zigzag：`-26.549%`
- Ulysses：`-27.155%`
- Ulysses 相对 Zigzag：`-0.825%`

同步后的 clean full-iteration mean 为 24.511 / 20.870 / 20.788 s；Ulysses 与 Zigzag 只差 0.395%，应视为基本打平。

### FA3 负载均衡

| 模式 | Critical-rank FA3 total / 2 iter | Max/min | Rank CV |
|---|---:|---:|---:|
| Contiguous | 18.993 s | 5.390x | 51.16% |
| Zigzag | 11.288 s | 1.003x | 0.09% |
| Ulysses | 11.418 s | 1.001x | 0.03% |

Zigzag 的 raw FA3 kernel time 略低于 Ulysses，但 Ulysses 的 CP communication path 更短，所以 model-only window 仍略快。

SAC 生效：FA3 forward kernel 数与 normal forward 的理论数量完全一致，backward 区间没有 FA3 forward replay。真正的 FA3 backward 仍保留。

### 两项实现级优化

Zigzag permutation cache：

- 首版每 2 iter 多出 226 次 `64-byte HtoD → cudaStreamSynchronize`。
- 修复后三种模式所有 rank 的 64-byte HtoD 均为 0。
- Zigzag model-only window 首版到修复后改善 4.177%；对应 control 只变化 -0.084%。
- `index_select` 的 DtoD copy 仍存在，但不再强制 drain CUDA stream。

Ulysses packed QKV：

- `ncclDevKernel_SendRecv`：672 → 336 次/2 iter。
- `ncclGroupEnd`：672 → 336 次/2 iter。
- `ncclSend` / `ncclRecv`：各 2,688 → 1,344 次/2 iter。
- Rank 0 `ncclGroupEnd` host duration：171.2 → 20.5 ms。
- 总传输字节基本不变；两轮端到端 A/B 没有测到可分辨的吞吐收益，说明 launch 减少被 pack/copy 或带宽主导成本抵消。

### 已排除的 blocker

- `rmsnorm_bwd_finalize_general_kernel` 约 **6.5 ms/iter**，不是当前 blocker。
- launch-to-start queue 仍可达到约 0.75–1.04 s，但 clean capture 的 all-kernel interval-union active 约 99%；这是 CPU 提前 enqueue 的队列深度，不是 host starvation，不应以缩短 queue 本身为优化目标。

## 后续方向

1. 用至少 20 个稳态 iterations、3 个 job-level repeats、交错顺序决定 Ulysses 与 Zigzag 的最终生产默认；当前约 1% 差距不足以做强结论。
2. Ulysses 的下一项 attention 优化应减少每层 12 个 FA3 prefix calls，例如原生 block-causal/multi-prefix kernel；当前总 launch 数最高。
3. Query-sharded 路径可继续尝试 packed K/V gather、缓存 `DeviceMesh`，以及避免 K/V chronological reorder copy。
4. 增加 CP8、跨节点 topology、partial final block、non-WORLD subgroup 和 SAC 参数梯度专项测试。
5. 若扩展到 multiview，需要单独设计 view/sequence layout 和 cross-view attention 通信；当前实现刻意拒绝该组合。

## 运行方式与产物

分支：

```text
perf/cp4-ulysses-zigzag-20260728
```

统一 launcher：

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 \
PROFILE_DATA_ROOT=/path/to/df93 \
bash samples/post-training/run_cp4_attention_ab.sh fa3-contiguous

CUDA_VISIBLE_DEVICES=1,2,3,4 \
PROFILE_DATA_ROOT=/path/to/df93 \
bash samples/post-training/run_cp4_attention_ab.sh fa3-zigzag

CUDA_VISIBLE_DEVICES=1,2,3,4 \
PROFILE_DATA_ROOT=/path/to/df93 \
bash samples/post-training/run_cp4_attention_ab.sh fa3-ulysses
```

四卡数值 oracle：

```bash
export OMNI_CACHE_DIR=/raid/john/.cache  # 或包含 FA3 venv 的其他 cache root
source samples/post-training/_env.sh
source samples/post-training/fa3_env.sh
cd post-training
CUDA_VISIBLE_DEVICES=1,2,3,4 \
"$OMNI_FA3_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 \
  ../samples/post-training/tests/torchrun_cp_attention_correctness.py
```

主要 artifacts：

- A/B：`/raid/john/.cache/nsys/omnidreams_df93_4gpu/cp4_attention/df93_cp4_fa3_*_ab*_20260728_*`
- Clean nsys：`/raid/john/.cache/nsys/omnidreams_df93_4gpu/cp4_attention/df93_cp4_fa3_*_nsys_opt_20260728_02`
- nsys 汇总：`/raid/john/.cache/nsys/omnidreams_df93_4gpu/cp4_attention/cp4_fa3_opt_nsys_comparison_20260728_02.analysis.md`

注意：nsys kernel lifetime 可包含 collective wait 且不同 stream 会 overlap，不能把所有 kernel duration 直接相加为 wall time。本报告使用 synchronized full iteration、跨 stage 的 model-only window，以及 per-rank attention kernel balance 共同判断。
