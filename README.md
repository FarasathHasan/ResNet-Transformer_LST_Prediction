# Urban Expansion Prediction using CA+ResNet+Transformer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning framework for predicting urban expansion patterns using a hybrid Cellular Automata (CA), Residual Networks (ResNet), and Transformer architecture. This model integrates CA principles with modern deep learning to simulate realistic urban growth scenarios.

---

## 🌟 Key Features

* **Hybrid Architecture**: CA + ResNet + Transformer
* **Memory-Efficient Spatial Attention**
* **Six Multi-Scenario Growth Simulations**
* **Realistic Growth Modeling with Controls & Constraints**
* **Full Evaluation Suite (Accuracy, F1, IoU, FoM, etc.)**
* **Attention Weight Extraction & Visualization**

---

## 📋 Table of Contents

* [Installation](#installation)
* [Dataset Structure](#dataset-structure)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Scenarios](#scenarios)
* [Evaluation Metrics](#evaluation-metrics)
* [Output Files](#output-files)
* [Advanced Configuration](#advanced-configuration)
* [Troubleshooting](#troubleshooting)
* [Dependencies](#dependencies)
* [Contributing](#contributing)
* [Citation](#citation)
* [License](#license)
* [Roadmap](#roadmap)
* [Performance Benchmarks](#performance-benchmarks)

---

## 🚀 Installation

### Prerequisites

* Python 3.8+
* CUDA GPU (Recommended: 8GB+ VRAM)
* GDAL installed on system

### Install System Dependencies

**Ubuntu/Debian**

```bash
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev python3-gdal
```

**macOS**

```bash
brew install gdal
```

**Windows**
Install from: *GISInternals*

---

### Install Python Dependencies

```bash
git clone https://github.com/yourusername/urban-expansion-prediction.git
cd urban-expansion-prediction

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## 📁 Dataset Structure

```
project_root/
│
├── DataColombo/
│   ├── 2015_cleaned.tif
│   ├── 2020_cleaned.tif
│   ├── 2025_cleaned.tif
│   ├── CBD_cleaned.tif
│   ├── road_cleaned.tif
│   ├── restricted_cleaned.tif
│   ├── pop_cleaned.tif
│   ├── slope_cleaned.tif
│   ├── amenitykernel_cleaned.tif
│   ├── commercialkernel_cleaned.tif
│   ├── industrailkernel_cleaned.tif
│   └── ntl_cleaned.tif
│
└── main.py
```

### Land Cover Classes

| Class | Meaning                  |
| ----- | ------------------------ |
| 0     | No data / Other          |
| 1     | Urban / Built-up         |
| 2     | Vegetation / Agriculture |
| 3     | Water                    |
| 4     | Open land                |

**All rasters must:**

* Match in size, CRS, resolution
* Be GeoTIFF
* Use same NoData value

---

## 💻 Usage

### Basic Example

```python
from main import LandCoverData, GrowthFactors, DeepLearningCA

landcover = LandCoverData(
    file1="DataColombo/2015_cleaned.tif",
    file2="DataColombo/2020_cleaned.tif",
    file3="DataColombo/2025_cleaned.tif"
)

factors = GrowthFactors(
    "DataColombo/CBD_cleaned.tif",
    "DataColombo/road_cleaned.tif",
    "DataColombo/pop_cleaned.tif",
    "DataColombo/slope_cleaned.tif",
    "DataColombo/restricted_cleaned.tif",
    "DataColombo/amenitykernel_cleaned.tif",
    "DataColombo/commercialkernel_cleaned.tif",
    "DataColombo/industrailkernel_cleaned.tif",
    "DataColombo/ntl_cleaned.tif"
)

model = DeepLearningCA(landcover, factors, patch_size=64)
model.build_model()

history = model.train(epochs=100, batch_size=8)

accuracy, f1, iou, predicted = model.evaluate(landcover.arr_lc3)

scenario_results = model.run_scenarios(
    start_year=landcover.arr_lc3,
    target_years=[2035, 2045]
)
```

### Command Line

```bash
python main.py
```

### Custom Scenario

```python
predictions, prob_maps = model.simulate_future(
    start_year=landcover.arr_lc3,
    target_years=[2030, 2040, 2050],
    scenario_name="custom",
    growth_multiplier=1.25
)
```

---

## 🏗️ Model Architecture

```
Input (10 rasters)  
     ↓
ResNet Blocks  
     ↓
Spatial Transformer  
     ↓
ResNet Blocks  
     ↓
Spatial Transformer  
     ↓
Growth Control Module  
     ↓
Projection  
     ↓
CA Constraints  
     ↓
Urban Expansion Output
```

### Components

#### **Residual Blocks**

* 3×3 conv
* BN + ReLU
* Skip connections

#### **Spatial Transformer**

* Multi-head attention (4 heads)
* Reduction ratio 4
* Positional encoding

#### **Growth Control Module**

* Global pooling + MLP

#### **CA Constraints**

* Neighborhood density
* Distance decay
* Restricted area mask
* Urban capacity limits

### Key Hyperparameters

| Parameter    | Value  |
| ------------ | ------ |
| Patch Size   | 64     |
| Batch Size   | 8      |
| LR           | 0.0005 |
| Weight Decay | 1e-4   |
| Epochs       | 100    |
| Early Stop   | 10     |
| Pos Weight   | 8      |
| Heads        | 4      |
| Reduction    | 4      |

---

## 🎯 Scenarios

| Scenario     | Growth | Meaning            |
| ------------ | ------ | ------------------ |
| Baseline     | 1.0    | Historic trend     |
| Slow         | 0.5    | Half-speed growth  |
| Moderate     | 0.75   | Controlled growth  |
| Accelerated  | 1.5    | Fast expansion     |
| Rapid        | 2.0    | Very rapid         |
| Conservation | 0.3    | Tight restrictions |

---

## 📊 Evaluation Metrics

### Classification

* Accuracy
* F1-Score
* IoU (Jaccard)
* Precision
* Recall

### Spatial Metrics

```
FoM = Hits / (Hits + Misses + False Alarms)
```

* Allocation Disagreement (AD)
* Quantity Disagreement (QD)

### Confusion Matrix

```
                Predicted
              NonUrban | Urban
Actual NonUrban   TN   |   FP
       Urban       FN   |   TP
```

---

## 📁 Output Files

### Checkpoints

* `best_model_transformer.pth`
* `final_model_transformer.pth`
* `checkpoint_epoch_X.pth`

### Predictions

* `predicted_2025_enhanced_transformer.tif`
* `predicted_YYYY_scenario.tif`

### Figures

* `training_history_enhanced_transformer.png`
* `comprehensive_scenario_comparison.png`

### Tables

* `scenario_summary.csv`

---

## 🔧 Advanced Configuration

### Memory Saving

```python
model.train(epochs=100, batch_size=4)
model = DeepLearningCA(landcover, factors, patch_size=32)
self.transformer1 = SpatialTransformer(reduction_ratio=8)
```

### Add New Factors

```python
factors = GrowthFactors("factor1.tif", "factor2.tif", "factorN.tif")
```

### Attention Extraction

```python
weights = model.model.get_attention_weights()
```

---

## 🐛 Troubleshooting

### CUDA OOM

* Reduce batch size
* Reduce patch size
* Increase reduction ratio

### GDAL Error

Install system GDAL, then:

```bash
pip install GDAL==$(gdal-config --version)
```

### Raster Size Mismatch

```bash
gdalwarp -tr 30 30 input.tif output.tif
```

### No Training Patches

* Ensure class `1` (urban) exists
* Reduce patch size
* Ensure temporal change exists

---

## 📚 Dependencies

* PyTorch
* GDAL
* NumPy
* scikit-learn
* matplotlib
* seaborn
* pandas
* scipy

---

## 🤝 Contributing

1. Fork repo
2. Create branch
3. Commit changes
4. Push and submit PR

Follow PEP8 + add docstrings + update docs.

---

## 📖 Citation

```bibtex
@software{urban_expansion_prediction,
  author = {Your Name},
  title = {Urban Expansion Prediction using CA+ResNet+Transformer},
  year = {2025},
  url = {https://github.com/yourusername/urban-expansion-prediction}
}
```

---

## 📄 License

MIT License.

---

## 🗺️ Roadmap

* [ ] Multi-class urban predictions
* [ ] Uncertainty estimation
* [ ] Real-time prediction
* [ ] Global dataset support
* [ ] Climate scenario integration
* [ ] Web dashboard
* [ ] Docker support
* [ ] Pre-trained weights

---

## 📈 Performance Benchmarks

| Task                 | Time    | Memory |
| -------------------- | ------- | ------ |
| 100 Epochs           | ~45 min | 18 GB  |
| 2048×2048 Prediction | ~15 sec | 8 GB   |
| 6 Scenarios          | ~3 min  | 10 GB  |

---

**Note**: This is research software under active development. Results should be validated against domain knowledge and local planning constraints before use in real-world applications.
