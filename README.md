# TabPFN + TabPFN-Extensions + LimiX 使用指南

本文档介绍如何在本项目中使用 TabPFN（含扩展库的事后集成 AutoTabPFN）与 LimiX，并提供可直接运行的分类/回归示例。

## 概览

- TabPFN：开箱即用的表格数据分类/回归模型。
- TabPFN-Extensions：提供增强功能与事后集成（AutoTabPFN），能组合多种 TabPFN 配置以获得更优性能。
- LimiX：针对分类/回归/缺失值插补的通用表格推理模型，支持“基于检索的集成推理”（需要高端 GPU），也支持无检索配置在普通设备上运行。

目录结构（节选）：
- `TabPFN/`：TabPFN 源码与脚本
- `tabpfn-extensions/`：扩展库源码
- `LimiX/`：LimiX 源码与推理脚本、配置
- `examples/`：示例脚本（分类/回归）
- `LimiX-16M.ckpt`：LimiX 已下载的模型文件（大文件，不建议推到 GitHub）

---

## 环境准备

建议 Python 3.10+，Linux 环境。

最小依赖（用于示例运行，CPU/单卡即可）：
```bash
pip install torch scikit-learn numpy pandas huggingface-hub tqdm einops scipy typing_extensions
```

TabPFN 与扩展库安装：
```bash
pip install -e ./TabPFN
```
```bash
pip install -e "./tabpfn-extensions[post_hoc_ensembles]"
```

可选：提前下载 TabPFN 权重（避免首次运行时自动下载）
```bash
python ./TabPFN/scripts/download_all_models.py
```

说明：
- `tabpfn-extensions[post_hoc_ensembles]` 会安装 AutoTabPFN 所需的 `autogluon.tabular`，用于事后集成。
- 如果只用扩展库的其他功能（不需要 AutoTabPFN），可以安装不带 extras 的扩展库。

---

## 运行示例：TabPFN（基础版）

二分类（乳腺癌数据）：
```bash
python examples/tabpfn_demo_classification.py
```

回归（Boston Housing）：
```bash
python examples/tabpfn_demo_regression.py
```

说明：
- 若你想提升稳定性，可在代码中将 `n_estimators` 增大（如 16/32），但会增加耗时。
- 默认会自动选择 GPU（若可用），否则使用 CPU。

---

## 运行示例：AutoTabPFN（事后集成，性能更优）

二分类：
```bash
python examples/auto_tabpfn_demo_classification.py
```

回归：
```bash
python examples/auto_tabpfn_demo_regression.py
```

说明：
- 首次运行会训练/集成多种 TabPFN 配置，耗时更长。可通过示例中的 `max_time` 与 `n_ensemble_models` 控制训练上限与模型数量。
- 如果出现 `ImportError: autogluon.tabular is not installed`，请确认已执行：
```bash
pip install -e "./tabpfn-extensions[post_hoc_ensembles]"
```

---

## 运行示例：LimiX（API 快速示例）

为了快速在普通设备上运行，推荐先使用“无检索”配置。

分类（无检索配置）：
```bash
python examples/limix_demo_classification.py
```

回归（无检索配置）：
```bash
python examples/limix_demo_regression.py
```

说明：
- `examples/limix_demo_*` 已使用本地模型文件路径 `./LimiX-16M.ckpt`。
- 配置文件使用 `LimiX/config/cls_default_noretrieval.json` 与 `LimiX/config/reg_default_noretrieval.json`，适合普通设备。
- 回归示例会对目标值进行标准化后评估 RMSE 与 R²。

---

## 使用 LimiX 仓库脚本（批量推理、支持 DDP）

分类（脚本强制要求 GPU）：
```bash
python LimiX/inference_classifier.py --inference_config_path ./LimiX/config/cls_default_noretrieval.json
```

多卡分布式（DDP）分类（需要多 GPU）：
```bash
torchrun --nproc_per_node=8 LimiX/inference_classifier.py --inference_with_DDP --inference_config_path ./LimiX/config/cls_default_retrieval.json
```

回归（脚本会自动选择 CPU/GPU）：
```bash
python LimiX/inference_regression.py --inference_config_path ./LimiX/config/reg_default_noretrieval.json
```

多卡分布式（DDP）回归：
```bash
torchrun --nproc_per_node=8 LimiX/inference_regression.py --inference_with_DDP --inference_config_path ./LimiX/config/reg_default_retrieval.json
```

说明：
- 如果 `--inference_config_path` 指定的路径不存在，脚本会自动生成默认配置（参见 `LimiX/utils/inference_utils.py` 的 `generate_infenerce_config`）。
- “基于检索的集成推理”（retrieval 配置）需要高端 GPU（≥RTX 4090），普通设备建议使用无检索配置。

---

## 常见问题与提示

- AutoTabPFN 导入报错：
  - 报错 `autogluon.tabular is not installed` 时，请用带 extras 的安装命令。
- 首次运行下载模型较慢：
  - 先执行 TabPFN 的下载脚本进行预下载。
- GPU／设备选择：
  - 示例中均默认自动使用 GPU（若可用），否则使用 CPU。
- 大模型文件（`.ckpt`）：
  - GitHub 单文件限制 100MB，不建议直接推到 GitHub。请在 `.gitignore` 中忽略（本项目已包含），或使用 HuggingFace 进行下载。

---

## 附：示例脚本列表

- TabPFN 基础版
  - `examples/tabpfn_demo_classification.py`
  - `examples/tabpfn_demo_regression.py`
- AutoTabPFN 事后集成
  - `examples/auto_tabpfn_demo_classification.py`
  - `examples/auto_tabpfn_demo_regression.py`
- LimiX API（无检索配置）
  - `examples/limix_demo_classification.py`
  - `examples/limix_demo_regression.py`

如需按你的机器资源（CPU/GPU 数量、内存）进一步调优（如缩短 `max_time`、降低 `n_ensemble_models`、切换配置），请联系维护者或根据需求调整示例中的参数。