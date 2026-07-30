# OmniDreams 训练性能优化简版汇总

**更新时间：** 2026-07-30

**分支：** `agent/retained-training-optimizations-20260730`

**范围：** 已验证有明显收益的 FA3、attention-output SAC、Zigzag/Ulysses
Context Parallel 和 Frame-level AdaLN。FA4、FC2-selective SAC、custom
prefix backward 和 Flex dense-prefix 对照不在本分支。

## 1. 结论

当前建议保留的优化如下：

| 优化 | 严格对照 | Iter 降时 | 吞吐提升 | 主要代价 |
|---|---:|---:|---:|---|
| FA3，CP=1 | `61.983 → 55.353 s` | **10.70%** | **11.98%** | `+0.315 GiB` allocated |
| Attention-output SAC，CP=1 | `55.353 → 48.603 s` | **12.19%** | **13.89%** | `+17.020 GiB/rank` allocated |
| Frame-level AdaLN，CP=1（A/B control 含 custom prefix） | `48.330 → 44.523 s` | **7.88%** | **8.55%** | allocated 反而 `-2.175 GiB` |
| Zigzag，CP=4 | `25.823 → 22.311 s` | **13.60%** | **15.74%** | layout/reorder 通信 |
| Ulysses，CP=4 | `25.823 → 22.088 s` | **14.47%** | **16.91%** | all-to-all，要求 heads 可被 CP 整除 |

CP=1 的历史参考端点为 `Flex + whole-block checkpoint 61.983 s` 到
`FA3 + SAC + custom prefix + Frame-level AdaLN 44.523 s`，参考降时
**28.17%**、吞吐提高 **39.22%**。其中 custom-prefix 仅贡献约 0.59%，
该低收益私有 ABI 路径没有进入本精简分支。因此
`61.983 → 44.523 s` 只用于说明历史优化上限，不是当前分支的一次严格
端到端复测。

CP=1 与 CP=4 的 global batch 和并行拓扑不同，两组绝对时间和百分比不能
直接相乘或相加。

## 2. 各项优化做了什么

### 2.1 FlashAttention-3

原 FlexAttention 用一个 BlockMask 表达 block causal attention。FA3 不接收
PyTorch Flex `BlockMask`，因此新实现按 temporal query block 拆分：

```text
Q0 attends K/V[0:0]
Q1 attends K/V[0:1]
...
Q11 attends K/V[0:11]
```

每个合法 Q/K block pair 只计算一次。CP=1、query-sharded CP 和 Ulysses
共用同一精确语义。

主要改动：

- `post-training/omnidreams/_src/omnidreams/modules/block_causal_flash_attention.py`
  增加 full-sequence、query-sharded 和 Ulysses FA3 实现。
- `post-training/omnidreams/_src/omnidreams/networks/causal_cosmos.py`
  增加 `training_attention_backend=flex|flash_attn_3` 和训练 dispatch。
- HDMap 路径同步 backend/mask 逻辑；cross-view 对未支持组合 fail-fast。
- `fa3_env.sh` 固定已验证的 CUDA 12.8、Torch 2.10 和 FA3-NV 1.0.3
  环境。

CP=1 正式 A/B 使用 4 × H20-3e、FSDP=4、93 帧、84,480 tokens：

```text
Flex + whole-block checkpoint   61.983 ± 0.110 s
FA3  + whole-block checkpoint   55.353 ± 0.118 s
```

主要收益来自 attention backward，而不是 prefix decomposition 本身。

### 2.2 Attention-output SAC

这里没有新增 checkpoint engine，而是让 causal FA3 使用仓库已有的
`predict2_2b_720_aggressive` policy：

```text
FA3 attention custom op  -> MUST_SAVE
其他算子                 -> PREFER_RECOMPUTE
```

因此 checkpoint 仍然存在，但保存 attention output，继续重算 QKV、MLP、
AdaLN 和 norm。93 帧配置每 iter 有：

```text
12 prefixes/layer × 28 layers = 336 FA3 outputs
```

nsys 显示 backward 中 attention forward replay 从 336 次降到 0；真正的
336 次 FA3 backward 仍保留。

```text
FA3 + whole-block checkpoint   55.353 s
FA3 + attention-output SAC     48.603 s
```

SAC 的交换关系约为：

```text
+17 GiB/rank allocated  <->  -6.75 s/iter
```

它适合有较大显存余量的 H20；不能直接假设同配置在 80-GiB GPU 上安全。

### 2.3 Zigzag / Ulysses Context Parallel

Contiguous CP 虽给每个 rank 相同 query 长度，但 causal query 越晚，
可访问的 K/V prefix 越长。CP=4 时四个 rank 的理论 block-pair 工作量为：

```text
6 : 15 : 24 : 33
```

两种均衡实现：

- **Zigzag**：把 sequence 切成 `2 × CP` 个 micro-chunks，每个 rank
  配对一个早期和一个晚期 chunk；继续使用 local-Q/full-KV。
- **Ulysses**：all-to-all 将 `[S/CP,H]` 转成 `[S,H/CP]`，每个 rank
  计算完整 sequence 的一部分 heads，再逆变换。

主要改动：

- `imaginaire/utils/context_parallel.py`：zigzag split、时序恢复和
  permutation index cache。
- `modules/ulysses_attention.py`：autograd-safe sequence/head all-to-all
  和 packed Q/K/V。
- `modules/block_causal_flash_attention.py`：query-sharded 与 Ulysses
  block-causal FA3。
- `run_cp_attention_ab.sh`：`NPROC`、`CP_SIZE`、`FSDP_SIZE` 均由用户设置，
  没有写死 CP=4 或四张 GPU。

CP=4 两轮反向顺序 A/B：

| Strategy | Mean | FA3 rank max/min |
|---|---:|---:|
| Contiguous | 25.823 s | 5.390x |
| Zigzag | 22.311 s | 1.003x |
| Ulysses | 22.088 s | 1.001x |

Ulysses 只比 Zigzag 快约 1%，当前证据不足以指定唯一默认：

- 当前 16-head、单机 NVLink 配置追求最高吞吐：优先 Ulysses。
- 需要避免 `num_heads % CP_SIZE == 0` 约束：使用 Zigzag。

### 2.4 Frame-level AdaLN

原实现先把每帧 timestep embedding 空间复制 `H×W` 次，再让每层三个
AdaLN-LoRA MLP 对相同的行重复执行 GEMM：

```text
[B,T,D] -> repeat -> [B,T×H×W,D] -> AdaLN MLP
```

优化利用 pointwise MLP 与 repeat 可交换：

```text
[B,T,D] -> AdaLN MLP -> repeat -> [B,T×H×W,3D]
```

`optimized_repeated_adaln.py` 将三个 `nn.Sequential` 包装为
`SpatiallyRepeatedSequential`，保留原子模块层级、parameter shape 和
`state_dict` key。已有 state-dict/FSDP checkpoint 可以加载。

同卡 A/B：

```text
FA3 + SAC + custom-prefix control   48.330 s
+ Frame-level AdaLN                 44.523 s
Peak allocated                 56.960 -> 54.785 GiB
```

这是实数数学上的等价变换，但 BF16 GEMM 的 reduction 顺序改变后不保证
bitwise gradient 一致。已有验证中 FP32 gradient relative-L2 约 `4e-7`；
BF16 gradient cosine 大于 `0.999989`，短程 loss 最大差 `1e-4`。

当前实现是构模前安装的 sample-side patch，只对 CP=1 和完整
frame-aligned chunk 生效；尚未接入 Zigzag/Ulysses。

## 3. 推荐使用方式

### CP=1：FA3 + SAC + AdaLN

下面是 profiling/A-B 入口，不会保存最终 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NPROC=4 CP_SIZE=1 FSDP_SIZE=4 \
  OMNI_OPTIMIZE_REPEATED_ADALN=1 \
  bash samples/post-training/run_fa3_attention_ab.sh fa3-sac
```

正式训练应在正常 training entry 中设置相同 backend/SAC 配置并保留
validation/checkpointer，不应直接复用 profiling entry。

### CP>1：FA3 + SAC + 均衡 CP

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NPROC=4 CP_SIZE=4 FSDP_SIZE=4 \
  bash samples/post-training/run_cp_attention_ab.sh fa3-ulysses
```

或：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NPROC=4 CP_SIZE=4 FSDP_SIZE=4 \
  bash samples/post-training/run_cp_attention_ab.sh fa3-zigzag
```

GPU 数和 CP/FSDP 值只是示例。Ulysses 要求 `num_heads % CP_SIZE == 0`；
Zigzag 要求 global sequence 可被 `2 × CP_SIZE` 整除。

## 4. 本分支包含与排除的内容

主要提交脉络：

- `4e4d595`：FA3 backend 和 FA3/SAC A/B 入口。
- `df83d2e`：Zigzag/Ulysses 核心实现。
- `632c774`、`7ad69cb`：CP 文档和可配置 topology。
- 本分支新增提交：仅提取此前验证过的 Frame-level AdaLN 实现、开关、
  回归测试和本汇总。

明确排除：

- FA4 exact：相对 FA3 + AdaLN 仅再降低 1.57%，且 research overlay 的
  protobuf 依赖与仓库安全基线冲突。
- Custom prefix backward：仅约 0.59%，依赖 FA3-NV 私有 backward ABI。
- FC2-selective SAC：约 1.2–1.3%，但 peak allocated 增加 8.379 GiB。
- Flex dense-prefix control：同卡测试比 Flex BlockMask 慢约 2.31%。

RMSNorm finalize 只有毫秒级；launch-to-start queue 是 GPU 满载形成的
pending backlog，也不作为优化方向。

## 5. 适用范围与验证说明

- 已验证：x86_64 Hopper/SM90、Python 3.13、CUDA 12.8、PyTorch 2.10、
  Transformer Engine 2.12、FA3-NV 1.0.3。
- 当前 FA3/CP 优化只接入 non-interleaved single-view causal training；
  multiview、cross-view、推理和 KV-cache 没有接入这些优化。
- 既有验证覆盖 FP16/BF16 dense oracle、Q/K/V gradients、四卡
  distributed oracle、SAC recompute、AdaLN state-dict 和短程训练 loss。
- 按本次整理要求，没有新增 benchmark 或测试运行；所有数字来自此前
  已完成的无 profiler A/B、nsys/NCU 报告和归档日志。

相关文档：

- [CP=4 Ulysses/Zigzag 实现与性能报告](./CP4_ULYSSES_ZIGZAG_REPORT.md)
- [可配置 Context Parallel 使用指南](./CONTEXT_PARALLEL_USAGE.md)
- [Post-training README](./README.md)
