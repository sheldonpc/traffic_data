# 时空交通数据预处理工具

*Read this in other languages: [English](README_EN.md)*

本项目提供了时空交通数据预处理和图网络构建工具，支持多种交通数据集的处理。

## 致谢

感谢 [FlashST](https://github.com/icecity96/FlashST-master) 项目提供的部分数据集和图处理方法。

## 项目结构

```
.
├── CA_District5/                   # 加州第5区交通数据
├── NYC_BIKE/                      # 纽约自行车数据
├── PEMS03/                        # PEMS03交通流数据
├── PEMS07M/                       # PEMS07M交通速度数据
├── chengdu_didi/                  # 成都滴滴出行数据
├── data_prepare/                  # 数据预处理模块
│   ├── data_conf/                 # 配置文件目录
│   │   ├── CA_District5_3dim_construct_12.conf
│   │   ├── METRLA_3dim_construct_12.conf
│   │   ├── NYC_BIKE_3dim_construct_12.conf
│   │   ├── PEMS03_3dim_construct_12.conf
│   │   ├── PEMS07M_3dim_construct_12.conf
│   │   ├── PEMS08_3dim_construct_12.conf
│   │   └── chengdu_didi_3dim_construct_12.conf
│   └── prepareData_STAEformer.py  # 数据预处理主程序
└── graph_process.py               # 图网络处理工具
```

## 主要功能

### 1. 数据预处理 (`data_prepare/prepareData_STAEformer.py`)

该模块用于构建模型训练所需的数据集，主要功能包括：

- **多数据集支持**：支持 PEMS03/04/07/08、NYC_BIKE、CA_District5、chengdu_didi 等多种交通数据集
- **时间特征嵌入**：自动添加日期、星期等时间维度特征
- **数据归一化**：对时间特征进行 Min-Max 归一化处理
- **数据分割**：按照配置文件自动分割训练集、验证集和测试集
- **序列构建**：根据历史步长和预测步长构建时序样本

### 2. 图网络处理 (`graph_process.py`)

该模块提供了丰富的交通网络图处理方法：

#### 图构建与加载
- 多种数据格式支持（CSV、NPY、PKL）
- 邻接矩阵和距离矩阵构建
- 权重矩阵计算

#### 图归一化方法
- **对称归一化**：`get_normalized_adj()` - 对称归一化邻接矩阵
- **非对称归一化**：`asym_adj()` - 行归一化处理
- **消息传递归一化**：`symmetric_message_passing_adj()` - 对称消息传递

#### 拉普拉斯矩阵计算
- **标准拉普拉斯**：`calculate_normalized_laplacian()`
- **对称归一化拉普拉斯**：`calculate_symmetric_normalized_laplacian()`
- **缩放拉普拉斯**：`calculate_scaled_laplacian()`
- **拉普拉斯位置编码**：`cal_lape()` - 用于图神经网络的位置编码

#### 转移矩阵
- **随机游走矩阵**：`transition_matrix()` - 用于图上随机游走
- **双向转移矩阵**：支持双向随机游走

## 使用方法

### 数据预处理

1. **配置数据集参数**：
   ```bash
   # 编辑对应的配置文件，例如：
   data_prepare/data_conf/PEMS08_3dim_construct_12.conf
   ```

   配置文件示例：
   ```ini
   [Data]
   num_of_vertices  = 170        # 节点数量
   time_slice_size = 5           # 时间片大小（分钟）
   train_ratio = 0.6             # 训练集比例
   val_ratio = 0.2               # 验证集比例
   test_ratio = 0.2              # 测试集比例
   data_file = ../data/PEMS08/data.npz  # 数据文件路径
   output_dir = ../data/PEMS08   # 输出目录
   
   [Training]
   num_his = 12                  # 历史步长
   num_pred = 12                 # 预测步长
   ```

2. **运行数据预处理**：
   ```bash
   cd data_prepare
   python prepareData_STAEformer.py --config ./data_conf/PEMS08_3dim_construct_12.conf
   ```

3. **输出结果**：
   - 生成包含训练集、验证集、测试集的 `.npz` 文件
   - 数据格式：`(样本数, 时间步长, 节点数, 特征维度)`
   - 包含原始特征 + 时间特征（日期、星期）

### 图网络处理

在您的代码中导入并使用图处理函数：

```python
from graph_process import *

# 加载并预处理图数据
args.dataset_graph = 'PEMS08'  # 设置数据集名称
pre_graph_dict(args)           # 预处理图数据

# 使用处理后的图数据
adj_matrix = args.A_dict['PEMS08']        # 归一化邻接矩阵
laplacian_pe = args.lpls_dict['PEMS08']   # 拉普拉斯位置编码
```

## 支持的数据集

| 数据集 | 类型 | 节点数 | 时间间隔 | 数据维度 |
|--------|------|--------|----------|----------|
| PEMS03 | 交通流 | 358 | 5分钟 | 流量 |
| PEMS04 | 交通流 | 307 | 5分钟 | 流量 |
| PEMS07 | 交通流 | 883 | 5分钟 | 流量 |
| PEMS08 | 交通流 | 170 | 5分钟 | 流量 |
| PEMS07M | 交通速度 | - | 5分钟 | 速度 |
| NYC_BIKE | 自行车需求 | - | 30分钟 | 需求量 |
| CA_District5 | 交通流 | - | 5分钟 | 流量 |
| chengdu_didi | 交通指数 | - | 10分钟 | 指数 |

## 核心特性

- ✅ 支持多种交通数据集格式
- ✅ 自动时间特征工程
- ✅ 灵活的数据分割配置
- ✅ 丰富的图处理算法
- ✅ 拉普拉斯位置编码支持
- ✅ 多种图归一化方法
- ✅ 易于扩展的模块化设计

## 依赖环境

```
numpy
pandas
scipy
torch
configparser
```

## 许可证

本项目部分代码和数据集来源于 FlashST 项目，请遵循相应的开源协议。