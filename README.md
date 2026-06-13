# 基于 BERT 的词性标注模型

本项目实现了一个基于预训练 BERT 模型的中文词性标注（Part-of-Speech Tagging）与序列标注系统，主要用于生成 PKU-SEGPOS 的 `span_list` 预测。系统通过命名实体识别（NER）中常用的 BIO 标注体系对中文文本进行词性追踪与分类。

## 项目特点
* **模型架构**：基于 Hugging Face 的 `bert-base-chinese` 预训练模型，并下游衔接 `BertForTokenClassification` 任务分类头。
* **标注体系**：采用 BIO（Begin, Inside, Outside）标注策略进行词性片段（Span）定位。
* **评估指标**：使用 `seqeval` 库对多标签序列标注结果进行精确度（Precision）、召回率（Recall）以及 F1 分数的综合评估。

---

## 依赖环境
运行本项目需要安装以下核心 Python 库：
```bash
pip install torch transformers seqeval
```

---
## 数据集
项目使用PKU-SEG数据集。该数据集基于2005年第二届国际汉语分词大赛（The Second International Chinese Word Segmentation Bakeoff）发布的PKU数据集，由人民日报新闻语料标注得到，在此基础上新增2000年1月和12月的人民日报语料所标注的数据，并重新对数据进行整合划分。同时项目对原存在乱码的train.jsonline进行了修正，确保数据的干净准确。
## 数据格式
模型输入与输出均采用 `.jsonline` 格式，每行代表一个独立的 JSON 对象。

### 输入/训练数据格式示例
```json
{
  "tokens": ["二", "○", "○", "○", "年", "贺", "词"], 
  "span_list": [
    {"start": 0, "end": 4, "type": "t"}, 
    {"start": 5, "end": 6, "type": "n"}
  ]
}
```
* `tokens`：经过初步切分或字符级拆分的 Token 列表。
* `span_list`：词性片段标注列表，包含标注起始索引 `start`、结束索引 `end` 以及具体的词性标签 `type`。

---

## 模型训练与超参数设置
项目在训练时使用了以下关键超参数与优化器配置：
* **训练轮数 (Epochs)**: 6
* **批次大小 (Batch Size)**: 32
* **最大序列长度 (Max Length)**: 128
* **基础学习率 (Learning Rate)**: 4e-5
* **权重衰减 (Weight Decay)**: 0.01
* **优化器**: AdamW（配置参数 `eps=1e-8`, `betas=(0.98, 0.989)`）
* **学习率调度器**: 带线性预热的调度器（Warmup 步数占总步数的 10%）
* **Dropout 概率**: Attention 机制 Dropout 为 0.09，隐藏层 Dropout 为 0.11

### 验证集表现 (Development Set Performance)
在 6 个 Epoch 的训练过程中，模型在验证集（Dev）上的各项指标迭代如下：

| Epoch | Train Loss | Dev F1 | Dev Precision | Dev Recall |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.8850 | 0.9408 | 0.9360 | 0.9455 |
| 2 | 0.1564 | 0.9620 | 0.9591 | 0.9650 |
| 3 | 0.1087 | 0.9718 | 0.9707 | 0.9730 |
| 4 | 0.0810 | 0.9790 | 0.9782 | 0.9797 |
| 5 | 0.0628 | 0.9834 | 0.9829 | 0.9840 |
| 6 | 0.0508 | 0.9857 | 0.9852 | 0.9862 |

*注：训练过程中性能最佳的模型和分词器将自动保存至指定模型目录（默认 `./model`）。*

---

## 支持的词性标签映射
模型内部维护了完整的中文词性及语素标签映射字典，具体对应关系如下：

| 标签缩写 | 含义说明 | 标签缩写 | 含义说明 |
| :--- | :--- | :--- | :--- |
| `n` | 名词 | `nr` | 人名 |
| `t` | 时间词 | `ns` | 地名 |
| `s` | 处所词 | `nt` | 团体机关单位名称 |
| `f` | 方位词 | `nz` | 其他专有名词 |
| `m` | 数词 | `Ng` | 名语素 |
| `q` | 量词 | `Vg` | 动语素 |
| `b` | 区别词 | `Ag` | 形容语素 |
| `r` | 代词 | `Tg` | 时语素 |
| `v` | 动词 | `Dg` | 副语素 |
| `a` | 形容词 | `vn` | 名动词 |
| `z` | 状态词 | `an` | 名形词 |
| `d` | 副词 | `vd` | 副动词 |
| `p` | 介词 | `ad` | 副形词 |
| `c` | 连词 | `dq` | 副词性语素 |
| `u` | 助词 | `g` | 语素 |
| `y` | 语气词 | `x` | 非语素字 |
| `e` | 叹词 | `w` | 标点符号 |
| `o` | 拟声词 | `i` | 成语 |
| `l` | 习用语 | `j` | 简称 |
| `h` | 前接成分 | `k` | 后接成分 |

---

## 运行与预测命令行参数
预测脚本支持通过命令行参数（`argparse`）进行动态配置。可配置的参数列表如下：

* `--input` : 需要预测的公开测试集文件路径（默认值：`test_public.jsonline`）。
* `--output` : 预测结果输出文件路径（默认值：`PKUSEGPOS.jsonline`）。
* `--train` : 训练集数据路径，主要用于辅助动态构建标签映射字典（默认值：`train.jsonline`）。
* `--model` : 训练完成后的预训练模型保存目录（默认值：`model`）。
* `--batch-size` : 推理时的批次大小（默认值：`32`）。
* `--max-len` : 文本序列的最大截断或填充长度（默认值：`128`）。

### 预测执行示例
在终端中直接使用如下命令运行预测脚本：
```bash
python predict.py --input test_public.jsonline --output PKUSEGPOS.jsonline --model ./model --batch-size 32
```
