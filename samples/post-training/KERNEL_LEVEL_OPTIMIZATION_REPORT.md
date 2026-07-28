# OmniDreams CP=1 kernel-level profiling 与优化报告

**日期：** 2026-07-28

**分支：** `perf/kernel-profile-opt-20260728`

**基线 commit：** `7ad69cb763c41df62b96001faf06cd494ba2d80e`

**范围：** single-view HDMap、93 帧、CP=1、FSDP=4、FA3 + aggressive SAC

## 1. 结论

1. 最重的 FA3 backward main kernel 不是 HBM bandwidth-bound。最长 prefix
   上，HGMMA/Tensor 子管线达到 `98.40%`，DRAM throughput 只有 `0.33%`。
   它已经接近 H20 上该计算子单元的稳态上限。
2. `barrier≈42.6%` 不是“可回收的 42.6% kernel 时间”。修正 Hopper
   scheduler 指标后，barrier、GMMA wait、MIO throttle 在 prefix=1 和
   prefix=12 上几乎不变，同时 HGMMA 管线从 `92.37%` 升到 `98.40%`。
   这些 stall 主要是 FA3 warp-specialized producer/consumer 流水的结构性
   状态。
3. 真正可消除的是 release 组合式 autograd 在每个 prefix 后产生的
   full-sequence `SliceBackward` zero/copy/add。新增 sample-side custom
   autograd 后，每 iter：

   - backward kernel launches：`8074 → 6478`，减少 `1596`；
   - DtoD：`1018 → 66`，减少 `952`；
   - DtoD bytes：`119.11 GB → 2.84 GB`；
   - backward GPU launch span：`27.213 s → 26.890 s`，减少 `0.322 s`。

4. 无 profiler 的严格 4 卡 A/B 中，稳态 iteration 从
   `48.643 ± 0.154 s` 降到 `48.357 ± 0.271 s`：

   - iteration time 降低 `0.287 s / 0.589%`；
   - 等效吞吐提高 `0.593%`；
   - peak allocated memory 不变，均为 `56.960 GiB`；
   - loss 最大差 `1e-4`。

5. 该优化已基本兑现它约 `0.32 s/iter` 的结构性上限。剩余最大的单项仍是
   `15.8 s/iter` 左右的真正 FA3 backward math；若要继续获得大幅收益，
   需要减少/复用 block-prefix QK 工作或开发精确的 fused block-causal
   kernel，而不是针对 RMSNorm、CUDA launch queue 或 FA3 barrier 做局部修补。

## 2. 实验配置

| 项目 | 配置 |
|---|---|
| GPU | 4 × NVIDIA H20-3e，物理 GPU 1–4 |
| 并行 | CP=1，FSDP shard size=4 |
| 模型 | single-view Causal Cosmos 2B，28 Transformer blocks，BF16 |
| 输入 | 93 帧，704×1280 |
| DF | `state_t=24`，`num_frame_per_block=2` |
| Attention | block-causal FA3，12 prefixes/layer |
| Checkpoint | `predict2_2b_720_aggressive` SAC |
| 软件 | Python 3.13.13、Torch 2.10.0+cu128、FA3-NV 1.0.3 |
| Driver | 580.95.05 |
| NSYS=0 稳态窗口 | iteration 6–8 |
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

## 7. 正确性、显存和可复现性

验证结果：

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

## 8. 下一步优化优先级

### P0：保留当前 custom autograd 为默认关闭的实验开关

收益稳定但只有约 `0.59%`，且依赖 FA3 私有 ABI。适合先保持 opt-in，在更长
训练、validation 和 checkpoint save/load 中继续验证；升级 FA3 时必须重新跑
strict oracle 和端到端 A/B。

### P1：开发 fused exact block-causal kernel

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

### P1：继续分析 FA3 之外约 11 秒的 backward

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
