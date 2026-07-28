# OmniDreams CP=1 kernel-level profiling 与优化报告

**日期：** 2026-07-28–29

**分支：** `perf/kernel-profile-opt-20260728`

**基线 commit：** `7ad69cb763c41df62b96001faf06cd494ba2d80e`

**范围：** single-view HDMap、93 帧、CP=1、FSDP=4、FA3/FA4 +
frame-level AdaLN + optional FC2-selective SAC

## 1. 结论

1. 最重的 FA3 backward main kernel 不是 HBM bandwidth-bound。最长 prefix
   上 HGMMA/Tensor 子管线达到 `98.40%`，DRAM throughput 只有 `0.33%`。
   `barrier≈42.6%` 是 warp-specialized pipeline 的结构状态，不是可直接
   回收的 wall time。
2. custom ragged-prefix autograd 删除了 release `SliceBackward` 的
   full-sequence zero/copy/add：backward launches `8074→6478`、
   DtoD `1018→66`，GPU span 减少 `0.322 s`，同卡端到端约 `0.59%`。
3. 严格归因后，custom 路径 backward critical stream 由
   `15.833 s FA3 + 11.021 s 非 FA3` 构成。非 FA3 的 `81.85%` 是两类
   GEMM：`6.056 s` BF16 NVJET 主干与 `3.069 s` FP32 AdaLN-LoRA。
4. NCU 证明 BF16 NVJET 已约 `99.1%` HGMMA/math SOL；FP32 AdaLN 约
   `91.1%` math SOL。前者需要减少 replay/调用结构，后者的根因是先把
   24 个 frame embeddings 空间复制 3520 倍再重复做 FP32 GEMM。
5. frame-level AdaLN 消除重复数学后，同卡稳态从
   `48.330±0.242 s` 降到 `44.523±0.233 s`：iteration time
   `-7.876%`、吞吐 `+8.550%`、peak allocated `-2.175 GiB`。
6. 官方 FA4 CuTeDSL 能用 128-token full-tile sparse metadata 精确表示当前
   12-block mask。一层从 optimized FA3 的 `812.750 ms` 降到
   `789.217 ms`；forward/backward metadata 互为转置，一次 fused
   forward/backward 取代 12 个 prefixes。
7. FC2-selective SAC 之前，`FA4 exact + frame-level AdaLN +
   aggressive SAC` 的 4-GPU 稳态为 `43.823±0.153 s`。相对同卡
   custom-prefix control：

   - absolute reduction：`4.507 s/iter`；
   - iteration time reduction：`9.325%`；
   - throughput improvement：`10.284%`；
   - peak allocated：`56.960→54.465 GiB`；
   - iteration 6–8 loss 最大差 `1e-4`。
8. 最终 NCU 显示 fused FA4 backward main 同样是 HGMMA compute-bound：
   Math SOL `98.35%`、DRAM `0.35%`。它仍占约 `15.658 s/iter`，下一步
   需要减少 attention 数学或搜索 FA4 schedule，而不是继续减少 launch。
   当前 FA4/CUTLASS metadata 与仓库 protobuf 7 安全要求冲突，因此实现
   强制使用 venv 外的 research overlay 和显式 opt-in，不属于
   production-ready 依赖。
9. 对 `6.056 s/iter` 的 BF16 NVJET 做源码/调用序列拆分后，确认
   `1.969 s` 是 SAC forward replay，`4.086 s` 才是真正的 linear
   backward。只保存主 MLP FC2 output 后，每层一个 replay 消失：
   NVJET `843→815 launches/iter`，replay `1.96895→1.40820 s`，
   true backward 保持 `4.085 s`。
10. protobuf 7.35.1 final overlay 的同卡 A/B 中，FC2-selective SAC 将
    clean iteration 7–8 从 `43.755` 降到 `43.185 s`
    (`-1.30%`，吞吐 `+1.32%`)，peak allocated
    `54.465→62.844 GiB`。若包含 iteration 6，收益为 `1.23%`；
    因此保守结论写作约 `1.2–1.3%`。
11. FA4 sparse-Q `640` 让 hd=128 backward 从 fallback `m64` 改用
    native `m80`，但 production-shape alternating microbenchmark
    反而由 `559.233` 增至 `564.142 ms` (`+0.88%`)；该候选已否决并从
    最终代码移除，固定 `128×128` metadata 不变。

RMSNorm finalize、NCCL 和 launch-to-start queue 都不是本轮主 blocker。
完整第二阶段归因、实现和验证见 §10–§15。

## 2. 实验配置

| 项目 | 配置 |
|---|---|
| GPU | 4 × NVIDIA H20-3e；初始 profile 为 1–4，第二阶段严格 A/B 为 1,2,3,6 |
| 并行 | CP=1，FSDP shard size=4 |
| 模型 | single-view Causal Cosmos 2B，28 Transformer blocks，BF16 |
| 输入 | 93 帧，704×1280 |
| DF | `state_t=24`，`num_frame_per_block=2` |
| Attention | block-causal FA3 12 prefixes/layer；或 fused exact FA4 1 sparse call/layer |
| Checkpoint | baseline 为 `predict2_2b_720_aggressive` SAC；候选只额外保存主 MLP FC2 output |
| 软件 | Python 3.13.13、Torch 2.10.0+cu128、FA3-NV 1.0.3、FA4 `g14c377950` |
| Driver | 580.95.05 |
| NSYS=0 稳态窗口 | 历史表使用 iteration 6–8；final overlay clean window 同时报告 6–8 与 7–8 |
| nsys capture | iteration 7–8，rank0 表格均按每 iter 归一化 |

NCU 使用用户提供的参考镜像：

```text
cosmos3-cu128-train-infer:20260720-dff6966446f2
image id: f0265ca206c3
CUDA: 12.8.1
NCU: 2025.1.1
```

宿主机设置了 `RmProfilingAdminOnly=1`，普通进程不能读取 performance
counters。本次通过：

```text
sg docker
docker run --gpus device=7 --cap-add SYS_ADMIN ...
```

完成计数器采集，没有修改宿主驱动配置。

## 3. NCU：FA3 backward main kernel

### 3.1 Prefix sweep

生产 shape microbenchmark 固定：

```text
B=1
Q=7040
K/V=prefix×7040
heads=16
head_dim=128
dtype=BF16
causal=False
```

NCU 只捕获一次 `FlashAttnBwdSm90` main kernel：

| Prefix | Kernel | ms/prefix | SM throughput | DRAM throughput | Occupancy | Barrier |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.326 ms | 8.326 ms | 92.38% | 0.67% | 15.62% | 42.69% |
| 4 | 31.919 ms | 7.980 ms | 96.91% | 0.39% | 15.62% | 42.64% |
| 8 | 63.104 ms | 7.888 ms | 98.05% | 0.35% | 15.62% | 42.63% |
| 12 | 94.176 ms | 7.848 ms | 98.41% | 0.33% | 15.62% | 42.63% |

Kernel duration 和 thread instructions 基本随 prefix 线性增长；每增加一个
prefix，增量约为 `7.85–8.0 ms`。越长的 prefix 反而更接近稳态效率，未出现
随长度恶化的 barrier 或 memory behavior。

NCU 使用 kernel replay、base clock 和 cache control，因此绝对时间比训练
timeline 高约 8%；这里用它判断瓶颈类型，不替代端到端时间。

### 3.2 修正后的 Hopper 指标

最新 profiling skill 的默认 full metric list 中，下面两个名称无效：

```text
smsp__average_warps_active.avg.per_cycle_active
smsp__average_warps_eligible.avg.per_cycle_active
```

正确名称是：

```text
smsp__warps_active.avg.per_cycle_active
smsp__warps_eligible.avg.per_cycle_active
```

默认列表也遗漏了 Hopper 的 GMMA、MIO 和 scheduler idle 指标。本次在参考
镜像中先用 `ncu --query-metrics` 验证 metric ID，再重新采集：

| 指标 | Prefix 1 | Prefix 12 |
|---|---:|---:|
| Kernel duration | 8.328 ms | 94.179 ms |
| SM throughput | 92.37% | 98.40% |
| Compute-memory pipeline | 23.37% | 24.89% |
| DRAM throughput | 0.66% | 0.33% |
| HGMMA-family pipe | 92.37% | **98.40%** |
| Active warps / scheduler | 2.50 | 2.50 |
| Eligible warps / scheduler | 0.09 | 0.09 |
| Issue active | 8.17% | 8.17% |
| Barrier stall | 42.68% | 42.63% |
| GMMA wait | 21.87% | 21.91% |
| MIO throttle | 14.64% | 14.65% |
| Sleeping | 8.66% | 8.60% |
| Long scoreboard | 4.21% | 4.21% |
| Math-pipe throttle | 0.02% | 0.02% |
| Not selected | 0.16% | 0.16% |
| Selected | 3.27% | 3.27% |

Prefix 12 的补充 breakdown：

| 指标 | 值 |
|---|---:|
| `launch__waves_per_multiprocessor` | 135.38 |
| L2 throughput | 11.63% |
| HGMMA-family tensor-pipe active | 98.41% |
| Generic tensor-pipe active | 24.60% |
| GMMA instruction issue | 8.36% |
| Scheduler `issue_inst0` | 91.83% |

`issue_inst0=91.83%` 与 HGMMA active `98.41%` 并不冲突。Hopper HGMMA 是异步
warp-group 指令：warp 发射少量 GMMA 后，Tensor pipe 会在很多 cycle 内继续
执行。producer/consumer warps 在 barrier、GMMA completion 和 shared-memory
路径上等待时，底层 HGMMA 仍然可以保持满流水。

因此：

- `barrier 42.63%` 是 active warp state 的占比，不是 kernel wall time；
- 低 generic instruction issue 不能解释为 Tensor Core 空闲；
- HBM、L2 bandwidth 和 math-pipe throttle 都不是当前限制；
- 提高理论 occupancy 或删除同步不太可能在不重写 FA3 pipeline 的前提下带来
  收益。

现有证据已经足够完成 broad bottleneck classification，因此没有继续做
source-line deep NCU。当前 wheel 是预编译 cubin；若未来需要改 FA3 kernel
内部调度，应先用 line info 重编译，再做 SASS/PC sampling。

## 4. nsys：可消除的 autograd scaffold

### 4.1 FA3 本体没有变化

按 `demangledName` 分类；两边每 iter 都是
`28 layers × 12 prefixes = 336`：

| 每 iter | Baseline launches | Custom launches | Baseline | Custom | Delta |
|---|---:|---:|---:|---:|---:|
| FA3 forward main | 336 | 336 | 6661.185 ms | 6653.387 ms | -7.799 ms |
| FA3 backward main | 336 | 336 | 15799.435 ms | 15814.821 ms | +15.385 ms |
| Bwd preprocess | 336 | 336 | 9.724 ms | 9.580 ms | -0.144 ms |
| Bwd postprocess | 336 | 336 | 8.368 ms | 8.366 ms | -0.002 ms |

FA3 backward main 的 `+0.097%` 差异属于运行波动。custom optimization 没有
少算 attention、没有改变 mask，也没有让 FA3 kernel 本体变快。

### 4.2 被删除的工作

| 每 iter | Baseline | Custom | Delta |
|---|---:|---:|---:|
| Backward kernel launches | 8074 | 6478 | **-1596** |
| Summed backward kernel time | 27.255 s | 26.981 s | -0.274 s |
| Backward GPU launch span | 27.213 s | 26.890 s | **-0.322 s** |
| DtoD count | 1018 | 66 | **-952** |
| DtoD bytes | 119,109,178,695 B | 2,843,071,815 B | **-116,266,106,880 B** |
| DtoD GPU time | 64.892 ms | 1.498 ms | **-63.394 ms** |

结构性 kernel 归因：

| Kernel/iter | Baseline | Custom | Launch delta | Time delta |
|---|---:|---:|---:|---:|
| BF16 fill | 1004.5 | 52.5 | -952 | -97.537 ms |
| FP32 fill | 342 | 6 | -336 | -0.322 ms |
| BF16 add | 1651 | 1343 | -308 | -160.229 ms |

Launch delta 与 28 层、12 prefixes 精确对应：

```text
952 = 28 × (12 Q slices + 11 K prefixes + 11 V prefixes)
336 = 28 × 12 unused LSE gradients
308 = 28 × 11 Q full-sequence accumulation
```

四类 GPU 时间下降之和：

```text
63.394 + 97.537 + 0.322 + 160.229 = 321.482 ms
```

几乎等于 backward GPU span 的 `322.392 ms` 改善，机制归因闭环。

### 4.3 CUDA queue 的解释

| NVTX/iter | Baseline | Custom | Delta |
|---|---:|---:|---:|
| Iteration | 48.534 s | 48.133 s | -0.401 s |
| Forward | 17.133 s | 17.055 s | -0.078 s |
| Backward | 28.518 s | 27.246 s | -1.272 s |
| Optimizer full | 2.882 s | 3.831 s | +0.949 s |

Optimizer 的实际 GPU workload 没有变化：两边都是 684 kernels/iter、约
21.2 ms summed GPU time、约 39 ms span。custom backward 更早结束 enqueue，
原先归在 backward host range 尾部的 GPU queue-drain 转移到了 optimizer
range。因此：

- `backward -1.272 s` 不是独立的端到端收益；
- `optimizer +0.949 s` 也不是 optimizer kernel 回退；
- 真正可加的 GPU 收益约为 backward span 的 `0.322 s`；
- launch-to-start queue 仍然主要表示 GPU backlog，不是 GPU idle。

## 5. 实现

新增：

```text
samples/post-training/optimized_block_causal_flash_attention.py
```

release forward 对每个 query block 调一次 FA3：

```text
Q_i attends K/V_[0:i]
```

forward 和保存的 output/LSE 完全不变。custom backward 从最后一个 prefix
反向执行：

1. 最后一个 prefix 覆盖完整 K/V，FA3 直接写入最终 `grad_key/grad_value`；
2. 更早的 ragged prefixes 只分配实际 prefix 大小的临时 dK/dV，再原地累加
   到对应 final prefix；
3. dQ blocks 互不重叠，直接写入最终 dQ slice；
4. B>1 时 non-contiguous dQ view 使用连续临时 buffer 后 copy；
5. 不再让通用 `SliceBackward` 为每个 view 创建完整序列的 zero/copy tensor。

该实现使用 `once_differentiable`，不支持高阶梯度。它直接调用 FA3-NV 私有
`_flash_attn_backward`，因此运行时锁定并校验 1.0.3.x ABI。

安装器只 patch `causal_cosmos` 的 CP=1 binding，不修改 backend module，
不会隐式影响 query-sharded Contiguous/Zigzag 或 Ulysses。launcher 规则：

```text
OMNI_FA3_CUSTOM_PREFIX_GRAD=0|1
1 requires mode=fa3|fa3-sac
1 requires CP_SIZE=1
```

`run_cp_attention_ab.sh` 会显式设为 `0`，防止调用方 shell 中遗留的变量污染
CP 策略 A/B。

release tree `post-training/` 未做任何修改。

## 6. Microbenchmark 与端到端收益

### 6.1 单层 production-shape

| 单层 | Release | Custom | Delta |
|---|---:|---:|---:|
| Forward | 238.748 ms | 238.980 ms | +0.232 ms |
| Backward | 579.351 ms | 568.315 ms | **-11.036 ms** |
| Total | 818.099 ms | 807.294 ms | **-10.805 ms** |

按 28 层投影：

```text
11.036 ms × 28 = 0.309 s/iter
```

与 nsys 的 `0.322 s` backward GPU span 以及无 profiler 的
`0.287 s` iteration 改善一致。

单层 Torch profiler 还显示：

| 指标 | Release | Custom |
|---|---:|---:|
| Peak allocated | 3.823 GiB | 3.772 GiB |
| Peak reserved | 4.234 GiB | 3.832 GiB |
| `slice_backward` | 34 | 0 |
| zero kernels | 46 | 0 |
| add CUDA time | 8.074 ms | 2.873 ms |

曾测试用 `torch._foreach_add_` 合并 K/V 两个 ragged add。相同 production
shape 的五次交替 A/B：

```text
separate add_: 568.194 ± 0.100 ms backward
foreach add_:  569.714 ± 0.203 ms backward
```

`foreach` 慢 `1.520 ms/layer`，因此已回退，没有保留负收益改动。

### 6.2 4 卡 NSYS=0

| 配置 | Iter 6 / 7 / 8 | Mean ± stdev |
|---|---:|---:|
| FA3 + SAC baseline | 48.82 / 48.57 / 48.54 s | **48.643 ± 0.154 s** |
| + custom prefix grad | 48.67 / 48.20 / 48.20 s | **48.357 ± 0.271 s** |

严格 A/B：

```text
absolute reduction: 0.287 s/iter
iteration time reduction: 0.589%
throughput improvement: 0.593%
```

相对同一 CUDA 12.8/Torch 2.10 环境中先前的
`Flex + whole-block checkpoint = 61.983 s` 参考基线，完整
`FA3 + SAC + custom prefix grad` 链路为：

```text
iteration time reduction: 21.98%
throughput improvement: 28.18%
```

其中当前 custom autograd 只贡献最后约 `0.59%`；主要收益仍来自 FA3 替换
FlexAttention 和 SAC 消除 attention forward replay。

用户最早提供的 `61.021 s` 是 8 GPU、不同软件栈和 profiling 口径，只作为
历史症状，不与当前 4 GPU A/B 直接计算 speedup。

## 7. 第一阶段正确性、显存和可复现性

本节是 custom-prefix 第一阶段的验证快照；第二阶段完整口径见 §14。

- 完整 sample tests：`67 passed, 3 skipped`；
- Hopper custom-attention 文件：`16 passed`；
- release-vs-custom BF16 24-block 直接 A/B；
- B=2、partial final block 直接 A/B；
- output 逐元素一致；
- Q/K/V gradients 使用 `atol=rtol=5e-4`，实际 production-like A/B
  的 Q relative L2 为 `1.87e-6`，K/V 一致；
- SAC checkpoint compatibility 逐元素一致；
- installer scope 测试通过；
- Ruff、`bash -n`、`git diff --check` 通过；
- 训练 iteration 6–8 loss：
  `0.0432/0.0523/0.0690` vs `0.0432/0.0523/0.0689`；
- peak allocated 两边均为 `56.960 GiB`；
- peak reserved 两边 rank max 均为 `63.215 GiB`。

`run_fa3_attention_ab.sh` 现在还会：

- 校验 custom flag 和作用范围；
- 记录 `PROFILE_DATA_ROOT`、dataset symlink 和目标文件 SHA256；
- 记录 requested/effective custom 状态和尾部 Hydra arguments；
- 保存 `source-snapshot/` 与校验和；
- 在无 git metadata 的 source distribution 中安全退化为
  `git_head=unavailable`。

## 8. 第一阶段结束时的优先级

本节保留第一阶段作出第二阶段决策时的依据。前两项现已完成，最终状态见
§10–§14。

### P0：保留当前 custom autograd 为默认关闭的实验开关

收益稳定但只有约 `0.59%`，且依赖 FA3 私有 ABI。适合先保持 opt-in，在更长
训练、validation 和 checkpoint save/load 中继续验证；升级 FA3 时必须重新跑
strict oracle 和端到端 A/B。

### P1：开发 fused exact block-causal kernel（已完成，见 §13）

当前每层有 12 个 FA3 calls；所有 K/V prefixes 总 QK math 是主成本。高价值
方向是一次 launch 内实现 tile-level block-causal mask，并在相邻 query blocks
间复用 K/V load/scheduling。标准 token-causal FA3 不能替代它，因为同一个
temporal block 内需要双向 full attention。

这是有机会影响 `15.8 s/iter` FA3 backward 主体的方向，但属于新 kernel
开发，需要：

1. FP32 dense oracle；
2. output 与 dQ/dK/dV strict distributed oracle；
3. block 数、partial block、B>1、不同 head count/head dim coverage；
4. NCU HGMMA active、register/shared-memory、waves 和 end-to-end 验证。

### P1：继续分析 FA3 之外约 11 秒的 backward（已完成，见 §10–§12）

custom 后 backward GPU span 约 `26.89 s`，其中 FA3 backward main 约
`15.81 s`。下一轮应从 nsys 按总 GPU time 排名的 GEMM、communication 和
elementwise fusion 候选中选 top kernels，再逐个做 NCU，而不是继续深挖已经
接近 HGMMA 满流水的 FA3 kernel。可评估 FP8 GEMM、通信重叠和真正占时的
fusion opportunity。

### P2：CP>1 使用已验证的 Ulysses/Zigzag 路径

本优化只解决 CP=1 的 autograd scaffold。CP>1 的工作/通信分配应继续使用
独立的 Ulysses/Zigzag 实现和 A/B；不要把本开关静默套到 query-sharded
backend。

### 暂不优先

- 调整 `rmsnorm_bwd_finalize_general_kernel`：最新 trace 只有约
  `8.74 ms/iter`；
- 仅优化 launch-to-start queue：GPU 接近持续忙，queue 主要是 backlog；
- 盲目提高 FA3 occupancy：当前 233 KiB shared memory、168 registers/thread、
  1 CTA/SM，但 HGMMA 子管线已经达到约 98.4%；
- 直接“删除 barrier”：没有证据表明 barrier 使 Tensor pipe 产生 bubble。

## 9. Artifacts

| 内容 | 路径 |
|---|---|
| NCU prefix sweep | `/raid/john/.cache/nsys/omnidreams_df93_4gpu/ncu_skill_20260728/prefix{1,4,8,12}_container/` |
| Corrected prefix 1 | `/raid/john/.cache/nsys/omnidreams_df93_4gpu/ncu_skill_20260728/prefix1_hopper_corrected/ncu.csv` |
| Corrected prefix 12 | `/raid/john/.cache/nsys/omnidreams_df93_4gpu/ncu_skill_20260728/prefix12_hopper_corrected/ncu.csv` |
| Prefix 12 breakdown | `/raid/john/.cache/nsys/omnidreams_df93_4gpu/ncu_skill_20260728/prefix12_hopper_breakdown/ncu.csv` |
| NSYS=0 baseline | `/raid/john/.cache/nsys/omnidreams_df93_4gpu/kernel_opt/df93_kernelopt_baseline_20260728_01/` |
| NSYS=0 custom | `/raid/john/.cache/nsys/omnidreams_df93_4gpu/kernel_opt/df93_kernelopt_customgrad_20260728_01/` |
| nsys custom trace | `/raid/john/.cache/nsys/omnidreams_df93_4gpu/kernel_opt/df93_kernelopt_customgrad_nsys_20260728_01/` |
| nsys baseline SQLite | `/raid/john/.cache/nsys/omnidreams_df93_4gpu/fa3_ab/df93_t210_fa3_sac_nsys_20260724_01/timing_rank0.sqlite` |
| nsys custom SQLite | `/raid/john/.cache/nsys/omnidreams_df93_4gpu/kernel_opt/df93_kernelopt_customgrad_nsys_20260728_01/timing_rank0.sqlite` |

## 10. 第二阶段：剩余约 11 秒 backward 的严格归因

本节继续使用 custom-prefix trace 的 iteration 7/8。按 CUDA Runtime
correlation 把 launch 归到 backward NVTX，而不是按 kernel 实际开始时间
粗切。后者会把 forward 尾部已经排队、但延迟执行的 kernel 错算成 backward。

| 每 iter 均值 | 时间 |
|---|---:|
| Host backward NVTX | 27.246 s |
| Backward semantic GPU span | 26.890 s |
| 主计算流 FA3 | 15.833 s |
| 主计算流非 FA3 | **11.021 s** |
| 其他流 kernel sum | 0.127 s |

其他流主要是与主流重叠的 FSDP/NCCL，不能再完整加到 critical-path
优化上限。按互斥 family 对全部非 FA3 kernel sum `11.149 s` 分类：

| Family | ms/iter | 占非 FA3 |
|---|---:|---:|
| BF16 NVJET 主干 GEMM | 6,055.797 | 54.32% |
| FP32 AdaLN-LoRA GEMM | 3,069.334 | 27.53% |
| copy/cast、add/mul、cat、GELU/SiLU | 1,321.799 | 11.86% |
| FA2 cross-attention backward | 356.368 | 3.20% |
| RMSNorm、LayerNorm、RoPE | 218.081 | 1.96% |
| NCCL/FSDP | 126.066 | 1.13% |
| 其他 | 1.132 | 0.01% |

两类 GEMM 合计 `9.125 s/iter`，占 `81.85%`。其中：

- NVJET `843` launches/iter，`840 = 28 layers × 30`；
- FP32 AdaLN `515` launches/iter，主体
  `504 = 28 layers × 18`；
- RMSNorm finalize 只有约 `8.75 ms/iter`，再次确认不是 blocker。

NVJET 还能按模块拆成：

| 子模块 | Launch/iter | ms/iter |
|---|---:|---:|
| MLP `2048→8192→2048` | 168 | 3,407.367 |
| Self-attention Q/K/V/out | 336 | 约 1,762.4 |
| Cross-attention projections | 336 | 约 884.7 |
| Block 外 projection | 3 | 1.304 |

按每层固定调用序列和 CUDA Runtime correlation 继续拆分：

| NVJET 阶段 | Launch/iter | ms/iter |
|---|---:|---:|
| SAC forward replay | 280 | **1,968.954** |
| True linear backward | 560 | **4,085.873** |
| Block 外 projection | 3 | 1.351 |

每层前 10 个 NVJET 是 replay、后 20 个是真 backward。MLP FC1/FC2
replay 分别为 `564.352/561.430 ms/iter`。保存 FC2 output 的 raw BF16
payload 为：

```text
28 × 84480 × 2048 × 2 bytes = 9.0234375 GiB
```

而保存 FC1 output 需再增加 `36.09375 GiB`，FC1+FC2 合计
`45.11719 GiB`，却只多回收约 `0.564 s`，因此首个候选只保存 FC2。

完整 SQL、调用序列和源码映射保存在：

```text
/raid/john/.cache/nsys/omnidreams_df93_4gpu/kernel_opt/
  df93_kernelopt_customgrad_nsys_20260728_01/
  NON_FA3_BACKWARD_ANALYSIS.md
```

## 11. NCU：非 FA3 GEMM 的 kernel-level 结论

### 11.1 BF16 NVJET MLP

首轮 NCU 使用同量级、但 weight 未转置的 `NNN` proxy：

```text
nvjet_tst_192x192_64x3_1x2_h_bz_coopB_NNN
```

| 指标 | 值 |
|---|---:|
| NCU replay duration | 22.010 ms |
| Math SOL / HGMMA-family active | 约 99.1% |
| DRAM throughput | 3.80% |
| L2 throughput | 9.39% |
| Achieved occupancy | 14.84% |
| Long scoreboard | 23.73% |
| Barrier | 17.46% |

这是全 SM、Tensor Core compute-saturated 的大 GEMM。低 occupancy 与
warp-specialized persistent kernel 共存，不能推导出“提高 occupancy 就会
更快”。调 tile、减少 launch 或优化 HBM 都不是优先方向；需要减少 replay
或改变调用结构。

2026-07-29 又在参考镜像中用 production FC2 dispatcher layout
`[84480,8192] × [8192,2048]` 精确捕获 `TNN`：

```text
nvjet_tst_128x256_64x4_2x1_v_bz_coopA_TNN
```

| 指标 | Exact FC2 TNN |
|---|---:|
| NCU replay duration | 21.849 ms |
| Math SOL | **98.84%** |
| DRAM throughput | 4.82% |
| Achieved occupancy | 14.83% |
| Barrier / long scoreboard | 27.72% / 16.38% |

因此 NNN proxy 的核心判断得到 exact-layout 复核：BF16 FC2 本身已是
compute-bound。若保持相同 BF16 FLOP，换另一个 GEMM tile 的局部空间很小；
本轮实际收益来自 SAC 直接删除 28 次 FC2 replay，而不是让单次 NVJET
更快。

### 11.2 FP32 AdaLN-LoRA

生产代表 kernel：

```text
sm80_xmma_gemm_f32f32_..._ffma_...
```

| 指标 | 值 |
|---|---:|
| NCU replay duration | 11.658 ms |
| Math SOL | 91.06% |
| DRAM throughput | 18.12% |
| FMA pipe | 64.03% |
| Tensor pipe | 0.03% |
| Achieved occupancy | 24.87% |

它是 strict-FP32 FFMA compute-bound GEMM。真正的问题不是 kernel 没调好，
而是源码先把 `T=24` 的 timestep embedding 空间重复 `H×W=3520` 次，再让
28 层、每层三个 AdaLN-LoRA MLP 对完全相同的 rows 重复做 FP32 GEMM。

因此本轮没有微调这两个现成 GEMM kernel。NCU 指向的正确行动分别是：

- NVJET：减少 SAC replay 或做经过实测的 packed/fused 调用；
- AdaLN：直接消除 `3520×` 的重复数学。

NCU artifacts：

```text
/raid/john/.cache/nsys/omnidreams_df93_4gpu/kernel_opt/
  ncu_nonfa3_20260728/bf16_mlp_fwd_nvjet/
/raid/john/.cache/nsys/omnidreams_df93_4gpu/kernel_opt/
  ncu_nonfa3_20260728/fp32_adaln_second/
```

## 12. Frame-level repeated AdaLN 优化

新增：

```text
samples/post-training/optimized_repeated_adaln.py
```

点式 MLP 满足：

```text
MLP(repeat_interleave(frame_embedding, H×W))
= repeat_interleave(MLP(frame_embedding), H×W)
```

sample-side startup patch 把三个 `nn.Sequential` 替换成同层级子模块的
`SpatiallyRepeatedSequential`：

1. 每帧只取一个 embedding row；
2. 在 `[B,T,D]` 上执行 SiLU 和两个 AdaLN-LoRA Linear；
3. 输出再按空间 token 展开；
4. 参数层级和 state-dict key 完全不变。

作用范围刻意 fail-closed：

- 当前只对 CP=None/1 生效，launcher 要求 `CP_SIZE=1`；
- 必须在构造 model 前安装，属于 one-shot process startup patch；
- KV-cache path 的 `current_start/current_end` 必须按 latent-frame
  boundary 对齐；
- unsupported child module、序列不能整除空间因子时直接报错。

这是实数数学上的等价变换，不是 BF16 bitwise 等价。审查 microbenchmark
中，FP32 gradient relative-L2 约 `4e-7`；BF16 因 GEMM reduction 顺序不同：

| 项目 | BF16 relative-L2 |
|---|---:|
| Input gradient, no LoRA | 约 0.40% |
| Input gradient, LoRA | 约 0.46% |
| Weight gradient | 约 0.27%–0.37% |
| Gradient cosine | > 0.999989 |

完整 block、SAC recompute、单 rank FSDP+SAC BF16 和 state-dict key 均通过。
真实训练还需以 loss 和短程稳定性作为准入，不声称 bitwise exact。

### 12.1 同卡 4-GPU 端到端 A/B

两边都使用物理 GPU `1,2,3,6`、CP=1、FSDP=4、FA3+aggressive SAC 和
custom-prefix grad；只切换 repeated AdaLN。

| 配置 | Iter 6 / 7 / 8 | Mean |
|---|---:|---:|
| custom-prefix control | 48.61 / 48.19 / 48.19 s | **48.330 s** |
| + frame-level AdaLN | 44.79 / 44.42 / 44.36 s | **44.523 s** |

严格同卡收益：

```text
absolute reduction: 3.807 s/iter
iteration time reduction: 7.876%
throughput improvement: 8.550%
peak allocated: 56.960 → 54.785 GiB (-2.175 GiB)
```

Iteration 6–8 loss：

```text
control:   0.0432 / 0.0523 / 0.0690
optimized: 0.0432 / 0.0523 / 0.0689
```

该 `3.81 s` 端到端改善与旧 trace 中 `3.07 s` FP32 GEMM 加上被同时缩小的
SiLU、add/cast/cat 以及 forward 工作相符。

Artifacts：

```text
/raid/john/.cache/nsys/omnidreams_df93_4gpu/kernel_opt/
  df93_kernelopt_customgrad_samegpu_20260728_02/
/raid/john/.cache/nsys/omnidreams_df93_4gpu/kernel_opt/
  df93_kernelopt_customgrad_repeated_adaln_20260728_01/
```

## 13. Fused exact block-causal FA4

新增：

```text
samples/post-training/fa4_exact_block_causal_attention.py
samples/post-training/fa4_env.sh
samples/post-training/tests/test_fa4_exact_block_causal_attention.py
```

生产 geometry：

```text
sequence_length = 84480 = 660 × 128
tokens_per_block = 7040 = 55 × 128
logical_blocks = 12
```

因此每个 `128×128` score tile 对 logical block-causal mask 要么全保留，
要么全删除，不存在 tile 内 partial token mask。实现构造：

- forward metadata：每个 Q tile 可访问同一或更早 logical block 的 K tiles；
- backward metadata：上述稀疏关系的精确转置；
- metadata 用 batch/head 维度 `1×1` 广播，不按 head 复制；
- 一次 FA4 forward 和一次 FA4 backward 取代每层 12 个 FA3 prefixes；
- partial final logical block 或 `tokens_per_block % 128 != 0` 直接拒绝。

FA4 private API 被封在名含 `flash_attn` 的 `torch.library.custom_op` 中。
现有 aggressive SAC policy 因此把它标为 `MUST_SAVE`；GPU test 确认
checkpoint backward 不会第二次调用 FA4 forward。installer 只 patch
`causal_cosmos` 的 CP=1 binding，不触碰 query-sharded 或 Ulysses。

依赖固定为：

```text
FlashAttention source commit 14c377950125c70b7a9dabf9c561fca53715ac7d
flash-attn-4 0.0.1.dev1+g14c377950
nvidia-cutlass-dsl 4.6.0.dev0
wheel sha256 aca91bc290ca656fa740005350c197342946396c0b04138eab95ddeb7be4d0bb
```

FA4 不能直接作为正式 FA3 venv 的 resolver-valid 依赖：
`nvidia-cutlass-dsl-libs-base==4.6.0.dev0` 声明 `protobuf<7`，仓库则因
安全约束要求 `protobuf>=7.35,<8`。本轮最终方案保持 protobuf 7.35.1，
把 FA4/CUTLASS 包安装在 venv 外的独立 `--target` overlay，并显式接受
上游 metadata override；6 项 Hopper GPU 测试在该组合上通过，但它仍
只属于 research profiling，不是 production-ready 环境。禁止为 FA4 将
共享 venv 降到 protobuf 6.33.x。overlay 只增加 `flash_attn.cute`，不会
替换 FA2 root package 或独立的 FA3-NV package。

清理共享 venv 中所有 FA4/CUTLASS 残留后，production shape 单层 oracle
也在 protobuf 7.35.1 overlay 上复现：output/dQ/dK/dV relative-L2 与
§13.1 完全相同；单次计时 `813.993→794.467 ms`，incremental peak
`2.483→1.954 GiB`。该单次结果用于验证隔离路径，不替代 §13.1 的多次
稳态均值。

必须启用 persistent CuTeDSL cache；未命中 cache 的首次 JIT 在本机约
`4.9 s forward + 3.7 s backward`，不计入稳态性能。

### 13.1 Production-shape 单层 A/B

| 单层 | Optimized FA3 | Fused FA4 | Delta |
|---|---:|---:|---:|
| Forward | 240.695 ms | 225.677 ms | -15.019 ms |
| Backward | 572.055 ms | 563.541 ms | -8.514 ms |
| Total | 812.750 ms | **789.217 ms** | **-23.533 ms / -2.90%** |
| Incremental peak | 2.483 GiB | 1.954 GiB | -0.529 GiB |

与现有 optimized FA3 的 BF16 correctness：

| 项目 | Max abs | Relative L2 |
|---|---:|---:|
| Output | 4.88e-4 | 9.63e-4 |
| dQ | 9.77e-4 | 7.78e-4 |
| dK | 9.77e-4 | 3.616e-3 |
| dV | 9.77e-4 | 3.612e-3 |

测试覆盖 1/5/7/12/17 logical blocks、不同 tiles/block、production
metadata、dense FP32 oracle、head_dim 64/128、forward 和 dQ/dK/dV、SAC
以及 installer scope。

### 13.2 被否决的替代实现

- FA3 `attention_chunk` 是同 chunk block-diagonal，不是 prefix
  block-causal，而且当前 backward 不支持；
- varlen FA3 若复制 K/V prefix，会额外产生约 `8.3 GiB` 临时 gradient；
- NATTEN exact tiled mask 在 production shape 上约
  `559 ms forward + 2470 ms backward`，远慢于 FA3；
- FlexAttention 已经是 exact one-call 参考，但端到端明显慢于 FA3；
- legacy FA2 block-sparse path 不覆盖当前 BF16/head_dim 128 需求。
- FA4 sparse-Q `640` 可同时整除 forward `m128` 和 backward `m80`，
  但 6 次 alternating production-shape microbenchmark 中，backward
  `559.233→564.142 ms` (`+4.909 ms / +0.88%`)；更少 Q tiles 没有抵消
  `m80` 的 dQ layout/schedule 代价，因此实现已回退，保留 sparse-Q
  `128` / backward `m64`。

FA4 artifact：

```text
/raid/john/.cache/research/fa4-cache/fa4-artifacts/
  final_backend_production_microbench.json
  fa4_sparse_q_schedule_m80_rejected.json
```

### 13.3 同卡 4-GPU 端到端收益

FA4 组合使用物理 GPU `1,2,3,6`，其余配置与 §12.1 完全相同。CuTeDSL
production cache 在计时前单卡预热，steady-state 不含首次 JIT。

下表保留依赖隔离修复前的历史三段 A/B。当时同一 FA4 wheel/CUTLASS
代码直接安装在 shared venv，并使用 protobuf 6.33.5。最终 protobuf
7.35.1 true-overlay 已于 2026-07-29 在同一组物理 GPU `1,2,3,6`
完成 4-GPU NSYS=0 复测：iteration 7/8 为 `43.76/43.75 s`，
mean `43.755 s`。它与旧 shared-venv 对应 clean mean `43.735 s`
只差 `0.020 s`，说明隔离后的 runtime 性能等价。

| 配置 | Iter 6 / 7 / 8 | Mean ± sample stdev | Peak allocated |
|---|---:|---:|---:|
| FA3 custom-prefix | 48.61 / 48.19 / 48.19 s | 48.330 ± 0.242 s | 56.960 GiB |
| + frame-level AdaLN | 44.79 / 44.42 / 44.36 s | 44.523 ± 0.233 s | 54.785 GiB |
| FA4 exact + frame-level AdaLN | 44.00 / 43.73 / 43.74 s | **43.823 ± 0.153 s** | **54.465 GiB** |

FA4 相对 frame-level AdaLN：

```text
absolute reduction: 0.700 s/iter
iteration time reduction: 1.572%
throughput improvement: 1.597%
peak allocated: -0.320 GiB
```

两项第二阶段优化合计相对同卡 custom-prefix control：

```text
absolute reduction: 4.507 s/iter
iteration time reduction: 9.325%
throughput improvement: 10.284%
peak allocated: -2.495 GiB
```

相对同一 CUDA 12.8/Torch 2.10 环境中的
`Flex + whole-block checkpoint = 61.983 s` 参考基线，最佳完整链路：

```text
iteration time reduction: 29.298%
throughput improvement: 41.438%
```

最佳组合 iteration 6–8 loss：

```text
0.0432 / 0.0523 / 0.0690
```

相对 frame-level AdaLN 的 `0.0432/0.0523/0.0689` 最大差 `1e-4`。

Artifact：

```text
/raid/john/.cache/nsys/omnidreams_df93_4gpu/kernel_opt/
  df93_kernelopt_fa4_repeated_adaln_20260728_01/
```

## 14. 最终 nsys + NCU 闭环

### 14.1 Backward critical path

最终 profile 使用 `FA4 exact + frame-level AdaLN + aggressive SAC`，
统计 iteration 7/8。旧、新 capture 均为 4 张同型号 H20-3e，但旧 trace
使用物理 GPU `1,2,3,4`，新 trace 使用 `1,2,3,6`。因此本节的
host/iteration 数据是同配置、同 nsys 口径的近似 A/B；§12–13 的
非 profile 数据才是严格同卡 A/B。具体 kernel signature 的 launch/time
变化可以直接归因。本节原始 FA4 + AdaLN nsys 采集于前述
shared-venv/protobuf 6.33.5 环境；最终 protobuf 7 overlay 后续已经完成
4-GPU NSYS=0 与 nsys 复测，新增 FC2-selective SAC 结果见 §15。

| 指标 | 旧 custom-prefix FA3 | FA4 + frame AdaLN | Delta |
|---|---:|---:|---:|
| Iteration NVTX | 48.133 s | **43.788 s** | -4.344 s / -9.03% |
| Host backward | 27.246 s | **23.605 s** | -3.641 s / -13.36% |
| Semantic backward GPU span | 26.890 s | **23.739 s** | -3.152 s / -11.72% |
| Backward 主流 kernel sum | 26.854 s | **23.710 s** | -3.144 s / -11.71% |
| Backward launches | 6478 | **5526** | -952 / -14.70% |

FA4 对 launch fragmentation 的改善很大，但没有删除 exact attention
数学：

| Attention path | 旧 FA3 launch / ms | 新 FA4 launch / ms | 时间变化 |
|---|---:|---:|---:|
| Forward | 336 / 6653.387 | 28 / 6291.553 | -361.834 ms / -5.44% |
| Backward main | 336 / 15814.821 | 28 / 15657.509 | -157.312 ms |
| Backward pre + post | 672 / 17.946 | 56 / 16.340 | -1.606 ms |
| **Backward 合计** | **1008 / 15832.767** | **84 / 15673.849** | **-158.918 ms / -1.00%** |

因此 12 个 prefixes 融成一个 sparse schedule 后，rank 负载和 launch
数量改善不会按比例转换成 iter 收益；主要 QK/softmax/AV 与梯度 FLOP
仍然存在。

### 14.2 原约 11 秒非 attention backward 的去向

全流非 attention 从 `11.149 s` 降到 `8.156 s`，主流从
`11.021 s` 降到 `8.036 s`。主要 family：

| Kernel family | 新 ms/iter | 结论 |
|---|---:|---|
| BF16 `nvjet` 主干 GEMM | **6056.178** | 与旧值 6055.797 ms 基本完全相同；第二 blocker |
| copy/cast | 660.291 | repeated AdaLN materialization 增加约 174.7 ms |
| elementwise add | 367.055 | 较旧值 445.986 ms 下降 |
| FA2 cross-attention bwd | 356.286 | 与旧值 356.368 ms 相同 |
| repeat/broadcast grad reduction | 42.672 | frame-level 路径新增 |
| FP32 AdaLN GEMM | **6.674** | 从 3069.334 ms 下降 99.78% |
| SiLU | **0.511** | 从 69.384 ms 降低 |
| RMSNorm finalize | **8.838** | 仍不是 blocker |

这证明约 3 秒 backward 改善来自消除重复 AdaLN 数学，不是 kernel
重命名或时间分类误差。新路径新增的 direct copy、repeat reduction 和
fill 约 `0.23 s`，仍远小于移除的约 `3.1 s` FP32/SiLU 工作。

### 14.3 FA4 backward main 的 NCU 结论

对 production geometry 的
`FlashAttentionBackwardSm90` 做 `--set full`、39-pass NCU replay。
NCU replay 本身使单次时长变为 `614.033 ms`，不与 nsys 的稳态
`559.197 ms` 直接比较；counter 结论如下：

| NCU counter | 值 |
|---|---:|
| BF16 HGMMA / Math SOL | **98.35%** |
| DRAM throughput | **0.35%** |
| Memory throughput / L2 hit | 26.56% / 98.25% |
| Block / grid | 384 threads / 10560 blocks |
| Registers / thread | 168 |
| Dynamic shared memory / block | 198656 B |
| Theoretical / achieved occupancy | 18.75% / 15.62% |
| Active / eligible warps per scheduler | 2.499 / 0.115 |
| Issue slots busy | 10.73% |

NCU full raw counters中，每条 issued instruction 对应的主要 stalled-warps
ratio 是 barrier `9.69`、GMMA `5.08`、MIO throttle `3.10` 和 long
scoreboard `2.93`；这些值是 ratio，不是时间百分比。

`gpu-kernel-profiling` 的确定性规则将该 kernel 判定为
`compute_bound`：HGMMA 已接近峰值，而 DRAM 几乎未饱和。虽然 register
和约 199 KiB shared memory 将每 SM 限制为一个 block，不能仅凭低
occupancy 推导应减小 tile；当前 warp-specialized schedule 已把 BF16
HGMMA 拉到 98.35%，提高 occupancy 很可能以更差的 tensor-core
efficiency 为代价。下一项有效实验必须减少 attention 数学/指令，或比较
经过完整 A/B 的 FA4 tile/schedule 配置；HBM、Python dispatch、再减少
launch 都不是主要方向。

### 14.4 后续优化优先级

1. **Attention 算法或 FA4 schedule 搜索。** FA4 backward main 为
   `15.658 s/iter`，约占 backward 主流 `66%`。保持 exact mask 时可做
   Q-tile/warp-group/cluster 参数扫描；若训练目标允许，windowed history、
   lower precision/FP8 或减少可见 history 才能实质降低 FLOP。
   sparse-Q `640` / backward `m80` 已实测回退 `0.88%`，不是有效候选。
2. **继续减少 BF16 主干 GEMM 数学，而非局部调 NVJET。**
   FC2-selective SAC 已实测回收约 `0.56 s`、增加 `8.379 GiB` peak，
   详见 §15；FC1+FC2 需要约 `45.1 GiB` raw retention 才多回收约
   `0.564 s`，性价比不足，当前拒绝。剩余 `4.085 s` true backward
   包含 MLP dX/dW 与 self/cross projection 的多种 shape/layout；
   FP8/fused MLP、packed projection 或结构性共享是当前最高置信方向，
   但在排除所有 BF16 kernel/layout 调优前，仍需分别补抓 dominant
   `NTT/NTN/NNN` true-backward signature 的 exact-layout NCU。
3. **消除 AdaLN repeat materialization。** 让 scale/shift/gate consumer
   接受 frame-strided broadcast，或融合到 norm/gated residual；当前
   signature 上限先按约 `0.23 s/iter`。
4. **最后处理 FA2 cross-attention。** backward 只有 `0.356 s/iter`；
   RMSNorm finalize 只有 `8.84 ms/iter`，不再单独优化。

约 4 秒 launch-to-start 是 GPU queue backlog，不是 idle。插入
`cuda.synchronize()` 只会把等待移动到 host 并改变 range 归属；减少
critical-path GPU 工作才会缩短 queue。

### 14.5 Artifacts

```text
/raid/john/.cache/nsys/omnidreams_df93_4gpu/kernel_opt/
  df93_kernelopt_fa4_repeated_adaln_nsys_20260728_01/
    timing_rank0.sqlite
    FINAL_NSYS_ANALYSIS.md
    final_nsys_analysis.sql

/raid/john/.cache/research/fa4-cache/fa4-artifacts/
  final_backend_production_microbench.json
  overlay_proto7_production_microbench.json
  ncu_fa4_bwd_main.ncu-rep
  ncu_fa4_bwd_main/
    ncu-details.csv
    ncu-long.csv
    profile.json
    profile-summary.md
    profile-action-digest.md
```

最终回归：

```text
CPU suite: 93 passed, 3 skipped, 7 GPU tests deselected
Hopper GPU suite: 7 passed
```

GPU suite 包含 BF16/FP16、batch 1/2、head_dim 64/128、dense output 与
dQ/dK/dV oracle、aggressive SAC 不重算、installer scope，以及实际
`flash_attn.cute`/`cutlass` module path 必须位于独立 overlay 的断言。

## 15. FC2-selective SAC follow-up（2026-07-29）

### 15.1 原理与实现

Predict2-2B 主 MLP 是 `2048→8192→2048`。当前 gated residual：

```python
mlp_out = mlp(normed_x)
x = x + gate_mlp * mlp_out
```

使 checkpoint backward 必须 replay 到 FC2 output。新增 sample-side
policy 保留原 attention `MUST_SAVE` 规则，并额外匹配：

```text
aten.mm([M,8192], [8192,2048])
```

leading dimension `M` 不固定，因此 batch、sequence 和 CP degree 可变；
其余 GEMM 继续 recompute。开关 `OMNI_SAC_SAVE_MLP_FC2=1` 只允许
`fa3-sac`，并禁止尾部 Hydra 参数覆盖 SAC mode，避免 metadata 与实际
policy 不一致。release tree 和 checkpoint parameter keys 均未修改。

### 15.2 protobuf 7 final-overlay 同卡 A/B

两组均使用物理 GPU `1,2,3,6`、CP=1/FSDP=4、FA4 exact、frame-level
AdaLN、相同数据和 final-overlay。FA4 source snapshot SHA256 在 baseline、
candidate 和 nsys 三次运行中均为：

```text
266ead75beb9e141f5993f85f67b27378d58c6ec147645b55fb46aa05f9afb24
```

| 配置 | Iter 6 / 7 / 8 | Mean 6–8 | Clean mean 7–8 | Peak allocated |
|---|---:|---:|---:|---:|
| Aggressive SAC baseline | 44.77 / 43.76 / 43.75 s | 44.093 s | 43.755 s | 54.465 GiB |
| + save MLP FC2 | 44.28 / 43.20 / 43.17 s | **43.550 s** | **43.185 s** | 62.844 GiB |
| Delta | -0.49 / -0.56 / -0.58 s | **-0.543 s / -1.23%** | **-0.570 s / -1.30%** | +8.379 GiB |

对应吞吐提升为 `1.25%`（iter 6–8）或 `1.32%`（clean iter 7–8）。
第一次候选复跑的 clean mean 为 `43.190 s`，第二次为 `43.185 s`；
两次只差 `5 ms`。最终 iter 6–8 loss 均为
`0.0432/0.0523/0.0689`。

实测 memory 增量 `8.379 GiB` 略低于 raw payload `9.023 GiB`，说明
checkpoint storage 与其他 activation 峰值并非完全同一时刻存活；仍应把
该开关视为明确的显存换时间策略。

### 15.3 nsys：只删除 replay，不删除 true backward

候选为 final-overlay rank0 iteration 7/8；对照来自
`df93_kernelopt_fa4_repeated_adaln_nsys_20260728_01` 的
shared-venv/protobuf 6 trace。两者使用相同 GPU `1,2,3,6`、FA4 wheel、
模型和 capture 配置，但依赖打包和 sample-side overlay 校验代码不同，
所以 host/iteration 数字属于近似 nsys A/B。§13.3 已证明两种依赖打包的
NSYS=0 clean mean 只差 `0.020 s`；而下面精确少 28 个 NVJET launch、
true backward 不变的 kernel 证据可以直接归因于 FC2 policy。

| 指标 | FA4 + AdaLN baseline（shared venv） | + save MLP FC2（final overlay） | Delta |
|---|---:|---:|---:|
| Iteration NVTX | 43.788 s | 43.098 s | -0.690 s |
| Host backward | 23.605 s | 23.111 s | -0.494 s |
| Semantic backward GPU span | 23.739 s | 23.169 s | **-0.569 s** |
| Backward launches | 5526 | 5498 | **-28** |
| NVJET launches | 843 | 815 | **-28** |
| NVJET total | 6056.178 ms | 5494.550 ms | -561.628 ms |
| FA4 backward main | 15657.509 ms | 15660.667 ms | +3.158 ms |

NVJET 调用序列的精确阶段拆分：

| 阶段 | Baseline ms/iter | FC2-save ms/iter | Delta |
|---|---:|---:|---:|
| SAC forward replay | 1968.954 | **1408.203** | **-560.751** |
| True linear backward | 4085.873 | **4085.171** | -0.702 |
| Block 外 | 1.351 | 1.176 | -0.175 |

每层 replay 从 10 个 GEMM 变成 9 个；true backward 仍为每层 20 个。
四个 rank 都是 `815 NVJET launches/iter`，NVJET 总时长
`5.487–5.495 s/iter`。这证明收益来自缓存 FC2 output，没有漏掉真实
gradient 计算。约 4 秒 launch-to-start backlog 仍存在，但随着 critical
path 缩短而自然缩短；它依旧不是独立 idle hole。

### 15.4 Exact-layout NVJET NCU

使用提供的 CUDA 12.8 镜像、GPU 7 和 `--cap-add SYS_ADMIN`，按最新
`gpu-kernel-profiling` 的 `capture-ncu.sh --tier full` 显式 33-metric
列表捕获 production FC2 TNN；这不是 NCU 内建的 `--set full`。宿主机
直接 NCU 会因 `RmProfilingAdminOnly=1` 返回
`ERR_NVGPUCTRPERM`；容器采集未修改宿主配置。

| Counter | 值 |
|---|---:|
| Kernel | `nvjet_tst_128x256_64x4_2x1_v_bz_coopA_TNN` |
| NCU duration | 21.849 ms |
| Math SOL | **98.84%** |
| DRAM throughput | 4.82% |
| Achieved occupancy | 14.83% |
| Barrier | 27.72% |
| Long scoreboard | 16.38% |

该 exact-layout 结果确认 FC2 forward/replay NVJET 的局部 kernel 已接近 compute
ceiling。低 occupancy 是 warp-specialized persistent GEMM 的资源布局，
在 Math SOL `98.84%` 时不能单独当成优化目标。下一步若继续处理剩余
`4.085 s` true backward，应优先评估降低计算量的 FP8/TE fused MLP、
packed projections 或结构性共享。这个 profile 只覆盖 FC2 TNN，
不能代替 MLP dX/dW 和 self/cross projection 的 `NTT/NTN/NNN`
exact-layout NCU；补齐这些 signature 前，不应笼统排除所有 BF16
kernel/layout 调优。

### 15.5 Artifacts 与回归

```text
/raid/john/.cache/nsys/omnidreams_df93_4gpu/kernel_opt/
  df93_kernelopt_fa4_repeated_adaln_nsys_20260728_01/
  df93_kernelopt_fa4_adaln_proto7_baseline_20260729_02/
  df93_kernelopt_fa4_adaln_fc2sac_final_20260729_02/
  df93_kernelopt_fa4_adaln_fc2sac_nsys_20260729_03/
    timing_rank{0,1,2,3}.nsys-rep
    timing_rank{0,1,2,3}.sqlite

/raid/john/.cache/research/fa4-cache/fa4-artifacts/
  ncu_nvjet_fc2_tnn_exact/
    benchmark.py
    ncu.csv
    profile.json
    profile-summary.md
    profile-action-digest.md
  fa4_sparse_q_schedule_m80_rejected.json
```

最终定向回归：

```text
Targeted FC2/FA4 subset: 22 CPU/dispatcher + 7 Hopper = 29 passed
Full CPU suite: 93 passed, 3 skipped, 7 GPU tests deselected
4-GPU training: 8/8 iterations completed, finite matching loss
```
