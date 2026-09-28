# nano-hybrid

一个从零开始的配对实验：**全注意力 Transformer vs 3:1 GatedDeltaNet 混合架构**，15M 参数规模，Windows 原生训练与基准测试。

[English README](README.md)

![loss curves](docs/figures/loss_curves.png)

两个模型、全配对——相同数据顺序、相同 eval 子集、相同超参、TinyStories 上 4000 步、4096 词表 BPE：

- **gpt**：8 × 门控注意力块（RoPE、GQA、逐头 q/k RMSNorm、SwiGLU）
- **hybrid**：3:1 Gated DeltaNet + 门控注意力，Qwen3-Next 式布局

## 结论

3 个配对 seed，端到端验证：

| 指标 | gpt | hybrid |
|---|---|---|
| val loss（3 seeds，均值±标准差） | 1.868 ± 0.007 | **1.789 ± 0.009** |
| 驻留 decode cache @4096 ctx | 512 MB | **133 MB**（~3.9×） |
| 解码吞吐 | ~700 tok/s | ~700 tok/s |

1. **混合架构质量更好**——每个 seed、每个 eval 点都领先。
2. **混合架构省显存**——驻留 cache 斜率约为 1/4，恰好等于注意力层占比（8 层里 2 层 vs 8 层保留 KV 状态）。
3. **诚实的否定结果**：这个尺度下**没有解码速度优势**。把 `torch.cat` 逐拷贝换成预分配 KV buffer 后，两边都在 ~700 tok/s——之前那个戏剧性差距是拷贝 artifact，不是架构本质。带宽差异要到更大模型/更长上下文/fused kernel 才显形。
4. **诚实的代价**：hybrid 训练慢 ~30%——delta-rule 路径是纯 PyTorch，而 baseline 注意力走的是 fused SDPA。

![decode bench](docs/figures/decode_bench.png)

## 这不是什么

- 不是一个可用的语言模型。15M 参数 + TinyStories = 幼儿园水平故事机（语法成型、逻辑混乱）。
- 不是 fused kernel 性能的复现。全部纯 PyTorch——没 Triton、没 CUDA graph。
- 不声称长上下文质量：训练 ctx 只有 256 token，超出全是外推。

## 复现

```
pip install -r requirements.txt
python smoke_test.py                       # chunk/recurrent 等价 + decode-cache 一致性检查
python -m nanohybrid.prepare_data          # 下载 TinyStories，训练 4096-BPE 分词器（输出到 ./data）
python -m nanohybrid.train --arch gpt      # 或 hybrid；RTX 5070 Ti 上约 15 分钟/4000 步（输出到 ./ckpt）
python -m nanohybrid.bench_decode          # 吞吐 + 驻留 cache 曲线
python -m nanohybrid.plot_results          # 生成图片到 ./ckpt/figures/
python -m nanohybrid.generate --arch hybrid --prompt "Once upon a time" --tokens 150
```

`smoke_test.py` 是真断言：chunk-vs-recurrent 等价、prefill→decode cache 一致性、生成冒烟。

## 上游副作用

尝试把这个 hybrid 模型导出 ONNX 时暴露了 `torch.onnx` 的一个缺失转换器：`aten::linalg_solve_triangular`（Gated DeltaNet 分块预填充用到）。已上报并修复 → [microsoft/onnxscript#3055](https://github.com/microsoft/onnxscript/pull/3055)。

## 环境

测试于 Windows 11 + Python 3.12 + PyTorch 2.x（CUDA 13）+ RTX 5070 Ti（sm_120）。WSL2 下亦可运行。

## License

MIT
