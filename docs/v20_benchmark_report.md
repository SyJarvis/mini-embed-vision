# v2.0 Benchmark 报告：e5-small + EVA02 双语版

> 2026-10-26 | run-001_20261026_v20-e5small-eva02-bilingual-183k
> 训练：183k 双语数据（en_ratio=0.5, passage:/query: 前缀, mean pooling），4×RTX6000D，早停 E24（best E19, val R@1=0.4434）
> 评测：best.pth，四项 benchmark 全量（Flickr30k 5000 查询/1000 图；ImageNet 50k 验证集）

## 核心结果对比

### Flickr30k T2I 检索

| 指标 | v1.5 zh | **v2.0 zh** | v1.5 en | **v2.0 en** | Nomic(en) |
|------|---------|------------|---------|------------|-----------|
| R@1  | 0.1188 | 0.0880 | 0.0398 | **0.0842** | 0.5110 |
| R@5  | 0.2906 | 0.2316 | 0.1236 | **0.2274** | 0.7710 |
| R@10 | 0.3810 | 0.3302 | 0.1946 | **0.3304** | 0.8546 |
| R@100| 0.7698 | 0.7430 | 0.5830 | **0.7790** | — |
| nDCG@10 | 0.2366 | 0.1925 | 0.1046 | **0.1899** | 0.6781 |

### ImageNet zero-shot 分类（50k 验证集）

| 指标 | v1.5 zh | **v2.0 zh** | v1.5 en | **v2.0 en** |
|------|---------|------------|---------|------------|
| Top-1 | 0.0593 | 0.0361 | 0.0154 | **0.0291** |
| Top-5 | — | 0.1190 | — | 0.0971 |

## 结论

1. **核心假设成立**：文本塔确实是英文瓶颈。换 e5-small 后英文全线大涨——Flickr en R@5 +84%（0.1236→0.2274），nDCG@10 +82%，ImageNet en Top-1 +89%。
2. **首次实现中英平衡**：v2.0 的 zh/en 几乎持平（Flickr R@5: 0.2316 vs 0.2274），v1.5 时代 2.4 倍的语言鸿沟消除。
3. **代价：中文回落 20-40%**（Flickr zh R@5 -20%，ImageNet zh Top-1 -39%）。原因：e5-small（384d）的中文编码弱于 bge-base-zh（768d）。
4. **天花板仍是数据域**：两版与 Nomic（R@5 0.77）的差距（3 倍+）远大于两版互相的差距——ImageNet 物体中心描述 ≠ Flickr 场景描述的域鸿沟未被解决，数据扩展仍是最大杠杆。
5. 成功标准复盘：en R@5 目标 ≥0.30（实际 0.2274，方向对未达标）；zh 守住 ≥0.25（实际 0.2316，略破线）。

## 下一步选项

- A. 按原计划做 COCO 场景域数据扩展（预期同时抬升 v2.0 的 zh/en）
- B. e5-base（768d, 278M）替换 e5-small 消融：预期找回中文损失 + 英文更强，代价是文本塔 +160M 冻结参数
- C. A+B 并行（GPU 空闲）

## 复现

```bash
# 训练（已完成）
torchrun --nproc_per_node=4 train.py --text-model multilingual-e5-small \
  --vision-model eva02_base_patch14_224.mim_in22k --vision-backend timm \
  --data-dir data/imagenet-1k-bilingual --en-ratio 0.5 \
  --text-prefix "passage: " --text-pooling mean \
  --batch-size 64 --epochs 30 --patience 5 --lr 2e-5 --warmup-ratio 0.1 \
  --save-dir checkpoints_v20_e5small_eva02

# 评测（本次命令已存 benchmark/results/v20_*/
# 权重：checkpoints_v20_e5small_eva02/best.pth (E19, 822M)
```
