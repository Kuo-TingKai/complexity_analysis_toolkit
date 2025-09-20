# Complexity Analysis Toolkit

一個全面的複雜系統複雜度分析工具包，整合了資訊論、動力學和網路結構三大分析方法。

## 概述

本工具包提供了一套完整的複雜度分析方法，用於衡量和分析複雜系統的多個維度：

### 🔬 資訊論方法
- **Synergy (協同效應)**: 測量系統整體湧現的資訊
- **Phi (Φ)**: 整合資訊理論中的整合資訊量測
- **Multi-information (多元資訊)**: 系統組件間的總相關資訊

### 🌊 動力學方法
- **Multiscale Entropy (多尺度熵)**: 不同時間尺度的熵分析
- **Fractal Scaling (分形縮放)**: 自相似性和縮放特性
- **Criticality (臨界性)**: 系統臨界性和相變測量

### 🌐 網路結構方法
- **Modularity (模組化)**: 網路社群結構測量
- **Hypergraph Entropy (超圖熵)**: 超圖結構的熵測量
- **Network Complexity (網路複雜度)**: 額外的網路複雜度指標

## 安裝

### 使用 pip 安裝

```bash
pip install -r requirements.txt
```

或者使用 setup.py：

```bash
python setup.py install
```

### 依賴套件

- numpy >= 1.21.0
- scipy >= 1.7.0
- networkx >= 2.6.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- pandas >= 1.3.0
- nolds >= 0.6.0
- pyphi >= 2.0.0
- scikit-learn >= 1.0.0
- numba >= 0.56.0

## 快速開始

### 基本使用

```python
import numpy as np
from complexity_analysis import ComplexityAnalyzer

# 生成示例數據
data = np.random.randn(1000, 5)  # 1000個時間點，5個變數

# 初始化分析器
analyzer = ComplexityAnalyzer()

# 進行綜合複雜度分析
results = analyzer.analyze_multivariate(data)

# 查看結果
print(f"綜合複雜度分數: {results['composite_complexity']:.4f}")
print(f"協同效應: {results['information_theory']['synergy']:.4f}")
print(f"多尺度熵均值: {results['dynamics']['average']['mse_mean']:.4f}")
print(f"模組化: {results['network_structure']['modularity']:.4f}")
```

### 單變量時間序列分析

```python
# 分析單變量時間序列
univariate_data = np.random.randn(1000)
dynamics_results = analyzer.analyze_univariate(univariate_data)

print(f"DFA Alpha: {dynamics_results['dfa_alpha']:.4f}")
print(f"臨界性指數: {dynamics_results['criticality_index']:.4f}")
```

### 系統比較

```python
# 比較多個系統的複雜度
systems_data = [system1_data, system2_data, system3_data]
system_names = ['System_A', 'System_B', 'System_C']

comparison_df = analyzer.compare_systems(systems_data, system_names)
print(comparison_df)
```

## 範例

### 基本範例

運行基本範例來了解工具包的功能：

```bash
python examples/basic_example.py
```

這個範例會：
- 生成四種不同複雜度的合成系統（隨機、線性、耦合振盪器、混沌）
- 對每個系統進行全面的複雜度分析
- 創建視覺化結果圖表
- 顯示詳細的分析結果

### 進階範例

運行進階範例來進行更深入的分析：

```bash
python examples/advanced_example.py
```

這個範例會：
- 分析金融市場的不同狀態（牛市、熊市、橫盤、危機）
- 分析生態系統的不同穩定性狀態
- 比較不同系統的複雜度演變
- 創建綜合比較視覺化

## API 參考

### ComplexityAnalyzer

主要的複雜度分析類別。

#### 初始化參數

```python
ComplexityAnalyzer(
    info_binning_method='uniform',    # 資訊論離散化方法
    info_n_bins=10,                   # 離散化區間數
    dynamics_tolerance=0.2,           # 動力學容忍度
    dynamics_max_scale=20,            # 最大時間尺度
    network_n_clusters=None,          # 網路聚類數
    network_clustering_method='spectral'  # 聚類方法
)
```

#### 主要方法

##### `analyze_multivariate(data, **kwargs)`
對多元時間序列進行綜合複雜度分析。

**參數:**
- `data`: 多元時間序列數據 (時間點 × 變數)
- `scales`: 動力學分析的尺度列表
- `network_threshold`: 網路構建的閾值
- `network_method`: 網路邊權重計算方法
- `hypergraph_k`: 超圖分析的超邊大小

**返回:** 包含所有複雜度測量的字典

##### `analyze_univariate(data, scales=None)`
對單變量時間序列進行動力學分析。

##### `compare_systems(data_list, system_names=None, **kwargs)`
比較多個系統的複雜度。

##### `get_complexity_summary(results)`
生成複雜度分析的人類可讀摘要。

## 輸出解釋

### 資訊論指標

- **Synergy**: 協同效應值，衡量系統整體湧現的資訊
- **Phi (Φ)**: 整合資訊量，衡量系統的整合程度
- **Multi-information**: 多元資訊，衡量變數間的總相關性

### 動力學指標

- **MSE Mean**: 多尺度熵均值，衡量時間複雜度
- **DFA Alpha**: 去趨勢波動分析指數，衡量長期記憶性
- **Hurst Exponent**: Hurst指數，衡量時間序列的持續性
- **Criticality Index**: 臨界性指數，衡量系統的臨界狀態

### 網路結構指標

- **Modularity**: 模組化係數，衡量社群結構強度
- **Density**: 網路密度，衡量連接密度
- **Avg Clustering**: 平均聚類係數，衡量局部連接性
- **Hypergraph Entropy**: 超圖熵，衡量高階結構複雜度

### 綜合指標

- **Composite Complexity**: 綜合複雜度分數，整合所有維度的測量

## 應用領域

本工具包適用於以下領域的複雜度分析：

- **金融市場**: 市場狀態分析、風險評估
- **生態系統**: 生態穩定性分析、物種相互作用
- **神經科學**: 腦網路分析、神經活動複雜度
- **社會系統**: 社交網路分析、資訊傳播
- **物理系統**: 相變分析、臨界現象研究
- **工程系統**: 系統穩定性、故障預測

## 注意事項

1. **數據品質**: 確保輸入數據品質良好，避免過多缺失值或異常值
2. **參數調整**: 根據具體應用調整分析參數（如離散化區間數、時間尺度等）
3. **計算資源**: 大數據集可能需要較長計算時間，建議適當調整參數
4. **結果解釋**: 複雜度指標的解釋需要結合具體應用背景

## 貢獻

歡迎提交問題報告、功能請求或代碼貢獻。請確保：

1. 代碼符合現有風格
2. 添加適當的測試
3. 更新相關文檔

## 授權

本專案採用 MIT 授權條款。

## 引用

如果您在研究中使用了本工具包，請引用：

```bibtex
@software{complexity_analysis_toolkit,
  title={Complexity Analysis Toolkit},
  author={Kevin Ting-Kai Kuo},
  year={2025},
  url={https://github.com/Kuo-TingKai/complexity_analysis_toolkit}
}
```

## 更新日誌

### v1.0.0 (2024)
- 初始版本發布
- 實現資訊論、動力學和網路結構分析方法
- 提供基本和進階範例
- 完整的API文檔和說明
