# DeNo
[中国科学：信息科学'2026] 基于排他约束的去噪偏多标记学习方法

## 1. 项目简介

DeNo 主要面向偏多标记学习（Partial Multi-Label Learning, PML）场景。代码流程包含以下两部分：

- `generate.py`：根据已有标签数据生成或转换训练所需的 complementary labels 和 PML labels。
- `main.py`：读取数据、训练模型、评估结果，并把每轮实验记录保存到结果文件中。

## 2. 运行环境与依赖

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn

如果你使用的是 GPU 版本 PyTorch，请按你的 CUDA 版本安装对应的 torch。

## 4. 数据说明

### 4.1 真实世界数据

真实世界数据集直接读取已有的数据文件，不需要额外的人工合成流程。当前代码中使用的是以下数据集：如`YeastBP`等。这时会直接读取csv文件，其中真实世界数据的 `com_label` 和 `pml_label` 文件通常已经预先准备好。

### 4.2 合成数据

scene，yeast等多标记数据集，通过噪声参数控制噪声水平以此来形成偏多标记数据集，便于比较不同 noisy level 下模型的表现。当前代码通过参数 `--nl` 控制噪声等级，支持：

- `1.0`
- `1.5`
- `2.0`
- `2.5`

对应的数据文件命名规则为：

- `dataset_{nl}_com_label.csv`
- `dataset_{nl}_pml_label.csv`

例如 `scene_2_com_label.csv`、`scene_2_pml_label.csv`。

## 5. 运行

### 5.1 合成标签

在 `DeNo` 目录下运行：

```bash
python generate.py
```

### 5.2 运行训练

合成数据集
```bash
python main.py --dataset scene --model linear --nl 2.0
```
真实世界数据集
```bash
python main.py --dataset YeastBP --model linear 
```


## 6. 结果文件

训练后会在以下目录中生成结果：

- `result/`
- `experiment/`

文件名格式中会包含数据集名和噪声等级，便于区分不同实验。

---
## citation
<code data-enlighter-language="raw" class="EnlighterJSRAW"> </code>
