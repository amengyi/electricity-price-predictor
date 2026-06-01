# ⚡ 电价预测系统

基于Streamlit框架开发的电价预测程序，支持多种机器学习模型进行电价预测，并提供直观的可视化界面。

## 📋 功能特点

### 1. 数据处理
- 数据加载与清洗
- 特征工程（时间特征、节假日特征、统计特征）
- 数据预处理与标准化

### 2. 模型支持
- **随机森林** - 集成学习模型，适用于非线性关系
- **岭回归** - 线性回归模型，适合处理多重共线性
- **XGBoost** - 梯度提升树，高性能预测模型
- **LightGBM** - 轻量级梯度提升树，训练速度快

### 3. 可视化界面
- 数据概览与统计
- 电价趋势图
- 特征相关性分析
- 模型性能评估
- 预测结果展示

### 4. 预测功能
- 历史数据查询
- 未来电价预测（1-14天）
- 预测结果导出（CSV格式）
- 特征重要性分析

## 🚀 快速开始

### 环境要求
- Python 3.10+
- pip 20.0+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 本地运行

```bash
streamlit run app.py
```

访问 http://localhost:8501 查看应用。

## 📦 项目结构

```
forecast_system/
├── app.py              # Streamlit主应用
├── data_processor.py   # 数据处理模块
├── models.py           # 机器学习模型模块
├── requirements.txt    # 依赖列表
├── railway.toml        # Railway部署配置
└── README.md           # 项目文档
```

## 📊 模型原理

### 随机森林
随机森林是一种集成学习方法，通过构建多个决策树并汇总它们的预测结果来提高预测准确性。它能够处理高维数据，具有较好的抗过拟合能力。

### 岭回归
岭回归是一种正则化线性回归方法，通过在损失函数中加入L2正则化项来防止过拟合，适用于处理具有多重共线性的数据。

### XGBoost
XGBoost是一种高效的梯度提升树算法，支持并行计算，具有正则化项防止过拟合，在各种机器学习竞赛中表现出色。

### LightGBM
LightGBM是微软开发的轻量级梯度提升框架，采用直方图优化和按叶子生长策略，训练速度快且内存消耗低。

## 🔧 参数说明

### 随机森林参数
- `n_estimators`: 决策树数量（默认100）
- `max_depth`: 树的最大深度（默认10）
- `min_samples_split`: 分裂所需的最小样本数（默认2）

### XGBoost参数
- `n_estimators`: 树数量（默认100）
- `max_depth`: 树的最大深度（默认6）
- `learning_rate`: 学习率（默认0.1）

### LightGBM参数
- `n_estimators`: 树数量（默认100）
- `max_depth`: 树的最大深度（默认6）
- `learning_rate`: 学习率（默认0.1）

## 🚢 部署到Railway

### 步骤

1. **安装Railway CLI**
```bash
npm install -g @railway/cli
```

2. **登录Railway**
```bash
railway login
```

3. **初始化项目**
```bash
railway init
```

4. **部署应用**
```bash
railway up
```

5. **设置环境变量（可选）**
```bash
railway env set PYTHON_VERSION 3.11
```

### 配置说明

`railway.toml` 文件包含部署配置：
- `startCommand`: 启动命令
- `healthcheckPath`: 健康检查路径
- `healthcheckTimeout`: 健康检查超时时间

## 📈 性能评估指标

- **RMSE（均方根误差）**: 衡量预测值与实际值之间的平均偏差
- **R²（决定系数）**: 衡量模型对数据的拟合程度（0-1）
- **SMAPE（对称平均绝对百分比误差）**: 百分比误差指标，更适合比较不同尺度的数据

## 📝 使用说明

1. **数据概览**: 查看数据统计信息、趋势图和相关性矩阵
2. **模型训练**: 选择模型类型，调整参数，点击训练按钮
3. **预测结果**: 设置预测天数，点击预测按钮，查看结果和导出
4. **性能评估**: 查看模型在训练集和测试集上的表现

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！