# 大模型显存计算器

数据核验日期：2026-07-05  
入口文件：`index.html`

## 原始任务目标

本项目的目标是实现一个电脑 Web 版大模型显存计算器，服务于大模型部署规划、KV Cache 容量评估、并行方式下每卡显存估算和 PD 分离部署设计。后续维护者或新 agent 接手时，应优先保持以下产品目标和约束：

- 支持模型：`GLM-5.2`、`DeepSeek-V4-Flash`、`DeepSeek-V4-Pro`、`Kimi-K2.6`、`MiniMax M3`。
- 模型数据必须基于 Hugging Face 配置页、`config.json`、`model.safetensors.index.json` 和对应论文/推理实现文档调研，不应凭空编参数。
- README 需要记录每个模型的关键配置参数、attention 模式、KV Cache 计算方式、权重大小、可用/官方推荐的数据格式，以及对应来源。
- KV Cache 计算公式要参考对应论文或主流推理实现文档，不能手写臆造。需要覆盖 MLA/latent KV、DeepSeek V4 CSA/HCA、MiniMax MSA/GQA 等不同 attention/KV 结构。
- 支持硬件：中国大陆版 NVIDIA RTX PRO 6000D 和 NVIDIA H20；页面和 README 都要列出硬件参数，并明确公开披露数据的不确定性。
- 计算部署显存构成：模型权重、KV Cache、固定运行时预留。上下文长度需要同时支持滑条、文本框和常用预设，并支持并发量输入。
- 数据格式选择只能列出模型实际支持或官方/主流部署文档中出现过的格式；不能为了凑选项自行添加 FP16 等未确认格式。
- 页面应拆成两个计算器：
  - 计算器一：模型总量。不要选择硬件，也不要选择并行方式；只根据模型、格式、上下文长度、并发和固定运行时预留，计算模型侧权重、KV Cache、运行时预留和总显存，并用饼图等方式可视化。
  - 计算器二：硬件部署。选择硬件数量、硬件型号和并行方式，考虑 TP/PP/EP/CP/DP、KV 分片方式、专家权重分片比例等对每张卡显存占用的影响。
- 页面需要能根据显存占用评估卡数，并给出 PD 分离部署时 P 池、D 池和 P->D KV 传输的设计参考。
- 可以估算性能时再做；如果 TTFT、TPOT 这类绝对值无法从公开数据可靠推导，不应强行输出。当前实现采用 KV 写入、KV 传输、MiniMax MSA attention FLOPs 等可追溯下界。
- 需要根据披露的硬件数据估算不同卡上、不同卡之间的 KV Cache 传输和 Prefill 成本。
- 页面要偏电脑 Web 看板，不需要手机适配；应充分利用桌面宽屏空间。
- 页面上选择某个模型时，应显示该模型的计算公式、参数具体数值和参数中文含义。
- README 应保留任务目标、数据来源、公式、模型参数、硬件参数和工程口径，方便新 agent 或后续维护者理解项目意图。

这是一个无依赖的静态网页，分成两个计算器：

- 计算器一：模型总量。只选择模型、权重/KV 格式、上下文长度、并发和固定运行时预留，不选择硬件和并行方式；输出权重、KV Cache、运行时预留、总显存，并用饼图展示占比。
- 计算器二：硬件部署。沿用计算器一的模型场景，再选择硬件数量、NVIDIA RTX PRO 6000D 中国大陆版 / NVIDIA H20、TP/PP/EP/CP/DP、KV 分片方式等；输出每张卡分到的权重、KV、每卡运行时预留、总占用，并用堆叠条展示是否超过可用显存，以及并行方案用卡是否超过选定硬件数量。

当前版本按电脑 Web 看板设计：左侧输入分成两个独立展开的计算器卡片；右侧顶部是指标卡和可视化，长表格和公式放在下方区域，并保留窄窗口下的单列阅读布局。

## 重要口径

- 模型权重大小使用 Hugging Face `model.safetensors.index.json` 的 `metadata.total_size`，单位是 bytes。
- 模型结构参数使用 Hugging Face `config.json`，没有把未出现在官方配置或官方/主流推理文档里的 FP16 等格式加进选项。
- KV Cache 公式来自对应 attention 机制的论文或推理实现文档：MLA latent KV、DeepSeek V4 CSA/HCA、MiniMax MSA/GQA。
- 页面分为两个计算器：模型总量计算器只看 `weight + KV + runtime reserve` 的总量；硬件部署计算器会按 TP/PP/EP/CP/DP、KV 分片方式和每卡固定运行时预留估算每张卡上的显存。
- 硬件并行视角里的“专家权重 EP 可分片占比”和“非专家权重复制度”是工程估算参数。公开 config 不提供完整逐 tensor 权重归属，因此页面不把该部分伪装成绝对精确值。
- 模型参数表增加了中文含义列；公式区域同时提供排版后的公式和纯文本公式，方便核对。
- 数据可视化包括：模型总显存饼图、每卡显存堆叠条。
- RTX PRO 6000D 与 H20 的中国大陆披露参数没有完整公开 NVIDIA datasheet；页面中硬件值采用公开披露/行业常见口径，并在备注中标出不确定性。
- TTFT/TPOT 没有做绝对预测。页面只输出 KV 写入、P->D KV 传输、MiniMax MSA attention FLOPs 这类可追溯下界。

## 计算公式

### 总显存

```text
total_bytes = weight_bytes + kv_cache_bytes + overhead_bytes
overhead_bytes = runtime_reserve_gb * 1e9
usable_gpu_bytes = gpu_marketing_GB * 1e9 * utilization_ratio
rough_capacity_cards = ceil(total_bytes / usable_gpu_bytes)
```

页面默认 `utilization_ratio = 90%`、`runtime_reserve_gb = 8`。这部分代表 activations、CUDA graph、runtime/workspace buffer、通信 buffer 等推理引擎预留，不是模型配置参数。vLLM/SGLang 等引擎通常通过可用显存比例、KV cache pool 和 profile/logs 来校准这部分，不能简单视为随 `weight + KV` 线性增长的百分比。

### 硬件并行后每卡显存

页面把并行方式拆成：

- `TP`：Tensor Parallel。估算中会切非专家可分片权重，也可按选项切 KV。
- `PP`：Pipeline Parallel。估算中会按层切分权重和 KV。
- `EP`：Expert Parallel。估算中只切专家权重，不切 dense/shared/embedding/router 等非专家权重。
- `CP`：Context Parallel。估算中默认不切权重，只按 KV 分片选项切 KV。
- `DP`：Data Parallel 副本。估算中复制权重，但把总并发按副本数拆分。

硬件视角的核心公式：

```text
cards_per_replica = TP * PP * EP * CP
total_cards = cards_per_replica * DP
available_cards = hardware_count
card_count_check = total_cards <= available_cards

per_replica_concurrency = ceil(total_concurrency / DP)
replica_kv_bytes = KV(per_replica_concurrency)

expert_weight = weight_bytes * expert_share
non_expert_weight = weight_bytes - expert_weight
replicated_weight = non_expert_weight * replicated_share
non_expert_shardable = non_expert_weight - replicated_weight

per_gpu_weight =
  replicated_weight
  + non_expert_shardable / (TP * PP)
  + expert_weight / (TP * PP * EP)

per_gpu_kv =
  replica_kv_bytes / (PP * kv_shard_factor)

per_gpu_total =
  per_gpu_weight + per_gpu_kv + runtime_reserve_per_gpu
```

`hardware_count` 是页面里填写的硬件数量；实际参与摊显存的卡数仍由 `TP * PP * EP * CP * DP` 决定。如果 `available_cards` 大于 `total_cards`，多出的卡不会自动降低每卡显存，需要手动增大 TP/PP/EP/CP/DP 或改变并行方案。

`kv_shard_factor` 由页面选项决定：

```text
PP 层切分 + TP×CP 分片: kv_shard_factor = TP * CP
PP 层切分 + TP 分片:    kv_shard_factor = TP
PP 层切分 + CP 分片:    kv_shard_factor = CP
仅 PP 层切分:           kv_shard_factor = 1
```

注意：不同推理引擎对 embedding/lm_head、router、shared experts、MoE gate、attention projection、KV block metadata、prefix cache、CUDA graph buffer 的处理不同。硬件视角用于规划和压测前校准，不替代引擎级 memory profiler。

### 常规 GQA/MHA KV

MiniMax M3 的主 KV 使用 GQA 口径：

```text
main_kv_bytes = C * T * L * 2 * num_key_value_heads * head_dim * bytes_per_element
```

其中 `C` 是并发请求数，`T` 是每请求保留的 KV tokens，`L` 是层数。`2` 代表 K 和 V。

### MLA latent KV

GLM-5.2 和 Kimi-K2.6 使用 MLA/latent KV 口径：

```text
mla_kv_bytes = C * T * L * (kv_lora_rank + qk_rope_head_dim) * bytes_per_element
```

这对应推理实现中将 compressed KV latent 与 RoPE key 部分拼成每 token 每层一个 latent KV entry 的做法。

### DeepSeek V4 CSA/HCA KV

DeepSeek V4 使用 hybrid CSA/HCA compressed attention。页面读取每层 `compress_ratios`：

```text
ratio = 4:
  layer_kv = ceil(T / 4) * main_entry_bytes
           + ceil(T / 4) * index_entry_bytes

ratio = 128:
  layer_kv = ceil(T / 128) * main_entry_bytes

ratio = 0:
  layer_kv = min(T, sliding_window) * main_entry_bytes

deepseek_v4_kv_bytes = C * sum(layer_kv over model layers)
```

DeepSeek V4 论文说明 RoPE 维度以 BF16 存、其余 KV 以 FP8 存，indexer QK cache 可用 FP4。vLLM 的 DeepSeek V4 部署建议使用 FP8 KV 和 FP4 indexer cache，因此页面提供：

- `vLLM FP8 KV + FP4 indexer`：`main_entry_bytes = 512`，`index_entry_bytes = 64`
- `论文 mixed`：`main_entry_bytes = (512 - 64) * 1 + 64 * 2 = 576`，`index_entry_bytes = 128 * 0.5 = 64`
- `BF16 fallback`：`main_entry_bytes = 512 * 2 = 1024`，`index_entry_bytes = 128 * 2 = 256`

### MiniMax M3 MSA KV

MiniMax M3 使用 GQA 主 KV + sparse indexer side cache：

```text
main_kv_bytes = C * T * L * 2 * num_key_value_heads * head_dim * kv_bytes
indexer_bytes = C * T * sparse_layers * sparse_index_dim * index_bytes
minimax_m3_kv_bytes = main_kv_bytes + indexer_bytes
```

MiniMax MSA 论文给出的 attention FLOPs：

```text
F_GQA(N) = 2 * Hq * dh * N^2
F_MSA(N) = Hkv * didx * N^2 + 4 * Hq * dh * N * k * Bk
```

页面把 3 个 dense attention 层按 `F_GQA` 估算，把 57 个 sparse attention 层按 `F_MSA` 估算，只作为 attention FLOPs 下界，不含 MoE/MLP、投影、采样和调度开销。

### KV 写入和 P->D 传输下界

```text
kv_write_lower_bound_seconds = kv_bytes / memory_bandwidth_bytes_per_second
pd_transfer_lower_bound_seconds = kv_bytes / interconnect_bandwidth_bytes_per_second
```

这只是物理带宽下界，不含 kernel、调度、网络协议、序列化、反压、block 管理和跨节点抖动。

## 模型参数

### GLM-5.2

来源：[HF model card](https://huggingface.co/zai-org/GLM-5.2)、[GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8)、[Transformers GLM MoE DSA 文档](https://huggingface.co/docs/transformers/en/model_doc/glm_moe_dsa)

| 项 | 值 |
|---|---:|
| HF repo | `zai-org/GLM-5.2` / `zai-org/GLM-5.2-FP8` |
| architecture | `GlmMoeDsaForCausalLM` |
| model_type | `glm_moe_dsa` |
| dtype | `bfloat16` |
| BF16 weight total_size | 1,506,659,919,872 bytes |
| FP8 weight total_size | 755,617,140,416 bytes |
| attention | MLA latent KV + Dynamic Sparse Attention + IndexShare |
| hidden_size | 6144 |
| num_hidden_layers | 78 |
| num_attention_heads | 64 |
| num_key_value_heads | 64 |
| head_dim | 192 |
| kv_lora_rank | 512 |
| q_lora_rank | 2048 |
| qk_rope_head_dim | 64 |
| qk_nope_head_dim | 192 |
| v_head_dim | 256 |
| max_position_embeddings | 1,048,576 |
| vocab_size | 154,880 |
| intermediate_size | 12,288 |
| moe_intermediate_size | 2048 |
| n_routed_experts | 256 |
| n_shared_experts | 1 |
| num_experts_per_tok | 8 |
| first_k_dense_replace | 3 |
| index_topk | 2048 |
| index_head_dim | 128 |
| index_n_heads | 32 |
| num_nextn_predict_layers | 1 |
| mlp_layer_types | 78 层：3 dense + 75 sparse |
| indexer_types | 78 层：21 full + 57 shared |

KV 公式：

```text
KV = C * T * 78 * (512 + 64) * bytes_per_element
```

### DeepSeek-V4-Flash

来源：[HF model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)、[DeepSeek V4 paper](https://arxiv.org/abs/2606.19348)、[vLLM DeepSeek V4 blog](https://vllm.ai/blog/2026-04-24-deepseek-v4)

| 项 | 值 |
|---|---:|
| HF repo | `deepseek-ai/DeepSeek-V4-Flash` |
| architecture | `DeepseekV4ForCausalLM` |
| model_type | `deepseek_v4` |
| torch_dtype | `bfloat16` |
| official weight format | FP4 experts + FP8 mixed |
| weight total_size | 159,609,485,896 bytes |
| published params | 284B total / 13B activated |
| attention | Hybrid CSA/HCA compressed attention |
| hidden_size | 4096 |
| num_hidden_layers | 43 |
| num_attention_heads | 64 |
| num_key_value_heads | 1 |
| head_dim | 512 |
| q_lora_rank | 1024 |
| qk_rope_head_dim | 64 |
| max_position_embeddings | 1,048,576 |
| vocab_size | 129,280 |
| moe_intermediate_size | 2048 |
| n_routed_experts | 256 |
| n_shared_experts | 1 |
| num_experts_per_tok | 6 |
| sliding_window | 128 |
| index_topk | 512 |
| index_head_dim | 128 |
| index_n_heads | 64 |
| num_nextn_predict_layers | 1 |
| expert_dtype | `fp4` |
| compress_ratios | 43 层：2 local/window + 21 CSA(m=4) + 20 HCA(m=128)；另有 1 个 MTP extra ratio=0 |

`compress_ratios` used by calculator:

```text
[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4]
```

### DeepSeek-V4-Pro

来源：[HF model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)、[DeepSeek V4 paper](https://arxiv.org/abs/2606.19348)、[vLLM DeepSeek V4 blog](https://vllm.ai/blog/2026-04-24-deepseek-v4)

| 项 | 值 |
|---|---:|
| HF repo | `deepseek-ai/DeepSeek-V4-Pro` |
| architecture | `DeepseekV4ForCausalLM` |
| model_type | `deepseek_v4` |
| torch_dtype | `bfloat16` |
| official weight format | FP4 experts + FP8 mixed |
| weight total_size | 864,704,792,696 bytes |
| published params | 1.6T total / 49B activated |
| attention | Hybrid CSA/HCA compressed attention |
| hidden_size | 7168 |
| num_hidden_layers | 61 |
| num_attention_heads | 128 |
| num_key_value_heads | 1 |
| head_dim | 512 |
| q_lora_rank | 1536 |
| qk_rope_head_dim | 64 |
| max_position_embeddings | 1,048,576 |
| vocab_size | 129,280 |
| moe_intermediate_size | 3072 |
| n_routed_experts | 384 |
| n_shared_experts | 1 |
| num_experts_per_tok | 6 |
| sliding_window | 128 |
| index_topk | 1024 |
| index_head_dim | 128 |
| index_n_heads | 64 |
| num_nextn_predict_layers | 1 |
| expert_dtype | `fp4` |
| compress_ratios | 61 层：30 CSA(m=4) + 31 HCA(m=128)；另有 1 个 MTP extra ratio=0 |

`compress_ratios` used by calculator:

```text
[128,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4]
```

### Kimi-K2.6

来源：[HF model card](https://huggingface.co/moonshotai/Kimi-K2.6)、[HF deploy guidance](https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/docs/deploy_guidance.md)、[DeepSeek-V2 MLA paper](https://arxiv.org/abs/2405.04434)

| 项 | 值 |
|---|---:|
| HF repo | `moonshotai/Kimi-K2.6` |
| top architecture | `KimiK25ForConditionalGeneration` |
| top model_type | `kimi_k25` |
| text architecture | `DeepseekV3ForCausalLM` |
| text model_type | `kimi_k2` |
| dtype | `bfloat16` |
| official weight format | native INT4 compressed-tensors, group_size=32 |
| weight total_size | 595,148,192,736 bytes |
| published params | 1T total / 32B activated |
| attention | MLA latent KV |
| hidden_size | 7168 |
| num_hidden_layers | 61 |
| num_attention_heads | 64 |
| num_key_value_heads | 64 |
| kv_lora_rank | 512 |
| q_lora_rank | 1536 |
| qk_rope_head_dim | 64 |
| qk_nope_head_dim | 128 |
| v_head_dim | 128 |
| max_position_embeddings | 262,144 |
| vocab_size | 163,840 |
| intermediate_size | 18,432 |
| moe_intermediate_size | 2048 |
| n_routed_experts | 384 |
| n_shared_experts | 1 |
| num_experts_per_tok | 8 |
| first_k_dense_replace | 1 |
| num_nextn_predict_layers | 0 |
| quantization_config | `compressed-tensors`, `pack-quantized`, INT4, `kv_cache_scheme=null` |

KV 公式：

```text
KV = C * T * 61 * (512 + 64) * 2
```

因为 HF 配置中 `kv_cache_scheme=null`，页面没有把 INT4 权重量化扩展到 KV Cache。

### MiniMax M3

来源：[HF model card](https://huggingface.co/MiniMaxAI/MiniMax-M3)、[MiniMax MSA paper](https://arxiv.org/abs/2606.13392)、[vLLM MiniMax M3 docs](https://docs.vllm.ai/en/latest/api/vllm/models/minimax_m3/)

| 项 | 值 |
|---|---:|
| HF repo | `MiniMaxAI/MiniMax-M3` |
| top architecture | `MiniMaxM3SparseForConditionalGeneration` |
| text architecture | `MiniMaxM3SparseForCausalLM` |
| model_type | `minimax_m3_vl` |
| torch_dtype | `bfloat16` |
| official weight format | BF16 |
| weight total_size | 869,157,697,024 bytes |
| published params | ~428B total / ~23B activated |
| attention | MiniMax Sparse Attention + GQA KV + indexer side cache |
| hidden_size | 6144 |
| num_hidden_layers | 60 |
| num_attention_heads | 64 |
| num_key_value_heads | 4 |
| head_dim | 128 |
| max_position_embeddings | 1,048,576 |
| vocab_size | 200,064 |
| intermediate_size | 3072 |
| num_local_experts | 128 |
| n_shared_experts | 1 |
| num_experts_per_tok | 4 |
| num_mtp_modules | 7 |
| num_nextn_predict_layers | 1 |
| sparse_index_dim | 128 |
| sparse_num_index_heads | 4 |
| sparse_topk_blocks | 16 |
| sparse_block_size | 128 |
| sparse_init_block | 0 |
| sparse_local_block | 1 |
| sparse_attention_freq | 60 层：3 dense + 57 sparse |

KV 公式：

```text
KV = C * T * 60 * 2 * 4 * 128 * kv_bytes
   + C * T * 57 * 128 * index_bytes
```

## 硬件参数

| 硬件 | 显存 | 显存带宽 | 默认卡间链路 | 备注 |
|---|---:|---:|---:|---|
| NVIDIA RTX PRO 6000D 中国大陆版 | 84GB GDDR7 | 约 1,398GB/s | PCIe Gen5 x16 约 64GB/s | 公开披露值不完全一致；NVIDIA 未公开完整 6000D datasheet。页面采用保守带宽口径。 |
| NVIDIA H20 | 96GB HBM3 | 约 4.0TB/s | NVLink/NVSwitch 约 900GB/s | NVIDIA 开发者论坛上也能看到没有公开完整 H20 datasheet 的讨论；页面采用行业常见披露口径。 |

参考来源：

- [NVIDIA RTX PRO 6000 Blackwell Server Edition official page](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)
- [Tom's Hardware: RTX PRO 6000D China market reporting](https://www.tomshardware.com/tech-industry/semiconductors/why-nobody-is-buying-nvidia-6000d-in-china)
- [NVIDIA Developer Forum: H20 datasheet discussion](https://forums.developer.nvidia.com/t/do-we-have-manual-datasheet-for-nvidia-h20/303749)
- [GetDeploying NVIDIA H20 summary](https://getdeploying.com/gpus/nvidia-h20)
- [GetDeploying RTX PRO 6000D summary](https://getdeploying.com/gpus/nvidia-rtx-pro-6000d)

## 文件说明

- `index.html`：静态网页，包含全部模型/硬件数据和计算逻辑。
- `README.md`：调研记录、公式、参数表、来源和口径说明。

## 使用建议

1. 先选择模型和权重格式。
2. 调整 `KV tokens / 请求`、并发数、Prefill 微批。
3. 如果是 PD 分离部署，Decode 池通常按所有活跃请求的常驻 KV 估算，Prefill 池按 prefill micro-batch 的临时 KV 和 KV 传输下界估算。
4. 真正上机时用目标推理引擎压测校正：尤其是 TP/EP/PP 粒度、专家并行、paged KV block、prefix cache、chunked prefill、speculative decoding 和多节点通信。
