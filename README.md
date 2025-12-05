# Land Surface Temperature Prediction using ResNet+Transformer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced deep learning framework for predicting Land Surface Temperature (LST) using a hybrid ResNet-Transformer architecture with Squeeze-and-Excitation attention. This model combines multi-scale feature extraction, spatial transformers, and physical thermal constraints to simulate realistic future LST patterns with consideration for urban expansion and heat island effects.

## 🌟 Key Features

- **Advanced Hybrid Architecture**: ResNet + Spatial Transformer + Squeeze-and-Excitation blocks
- **Multi-Scale Feature Fusion**: Hierarchical processing with skip connections
- **Enhanced Feature Engineering**: 19+ derived features including urban intensity, vegetation cooling, and interaction terms
- **Physical Thermal Constraints**: Urban Heat Island (UHI) modeling, NDVI-based cooling, and spatial coherence
- **Urban Expansion Awareness**: Enhanced warming effects for future urban areas
- **Robust Evaluation**: RMSE, MAE, R², Bias, Skill Score, and UHI intensity metrics
- **Memory Efficient**: Optimized spatial attention with reduction mechanisms for large-scale predictions

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Input Features](#input-features)
- [Thermal Constraints](#thermal-constraints)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Files](#output-files)
- [Citation](#citation)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- GDAL library installed on your system

### Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev python3-gdal
```

**macOS (using Homebrew):**
```bash
brew install gdal
```

**Windows:**
Download and install GDAL from [GISInternals](https://www.gisinternals.com/)

### Install Python Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/lst-prediction.git
cd lst-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## 📁 Dataset Structure

Your data should be organized as follows:

```
project_root/
│
├── ForLST/
│   # LST Data (Kelvin or Celsius)
│   ├── LST2015.tif
│   ├── LST2020.tif
│   ├── LST2025.tif
│   
│   # Land Use Classification
│   ├── 2015_cleaned.tif
│   ├── 2020_cleaned.tif
│   ├── 2025_cleaned.tif
│   
│   # NDVI Data (-1 to 1)
│   ├── NDVI 2015.tif
│   ├── NDVI 2020.tif
│   ├── NDVI 2025.tif
│   
│   # Surface Emissivity (0 to 1)
│   ├── Emissivity 2015.tif
│   ├── Emissivity 2020.tif
│   ├── Emissivity 2025.tif
│   
│   # Socioeconomic and Environmental Factors
│   ├── amenitykernel_cleaned.tif      # Amenity density
│   ├── building density.tif            # Building footprint density
│   ├── CBD_cleaned (1).tif             # Distance to CBD
│   ├── commercialkernel_cleaned.tif    # Commercial area density
│   ├── industrailkernel_cleaned.tif    # Industrial area density
│   ├── ntl_cleaned.tif                 # Nighttime lights
│   ├── pop_cleaned.tif                 # Population density
│   ├── restricted_cleaned.tif          # Restricted development areas
│   ├── road density.tif                # Road network density
│   ├── road_cleaned.tif                # Distance to roads
│   └── slope_cleaned.tif               # Terrain slope
│   
│   # Future Land Use Predictions (from urban expansion model)
│   ├── predicted_2035_enhanced_transformer.tif
│   └── predicted_2045_enhanced_transformer.tif
│
└── main.py
```

### Data Requirements

**LST Data:**
- Format: GeoTIFF
- Unit: Celsius (°C) or Kelvin (K)
- Typical range: 15-50°C (288-323K)
- Temporal resolution: Multi-year (e.g., 2015, 2020, 2025)

**Land Use Classes:**
- `1`: Urban / Built-up
- `2`: Vegetation / Agricultural
- `3`: Water bodies
- `4`: Open land / Bare soil
- `0`: No data

**NDVI (Normalized Difference Vegetation Index):**
- Range: -1 to 1
- Water: < 0
- Bare soil: 0 to 0.2
- Vegetation: 0.2 to 1

**Emissivity:**
- Range: 0 to 1
- Typical values: 0.95-0.99 for most land surfaces

**All input rasters must:**
- Have the same spatial extent and resolution
- Be in GeoTIFF format
- Use the same coordinate reference system
- Handle NoData values appropriately

## 💻 Usage

### Basic Usage

```python
from main import LSTData, LSTFactors, LSTSimulator

# Load LST and related data
lst_data = LSTData(
    lst_files=["ForLST/LST2015.tif", "ForLST/LST2020.tif", "ForLST/LST2025.tif"],
    landuse_files=["ForLST/2015_cleaned.tif", "ForLST/2020_cleaned.tif", "ForLST/2025_cleaned.tif"],
    ndvi_files=["ForLST/NDVI 2015.tif", "ForLST/NDVI 2020.tif", "ForLST/NDVI 2025.tif"],
    emissivity_files=["ForLST/Emissivity 2015.tif", "ForLST/Emissivity 2020.tif", "ForLST/Emissivity 2025.tif"]
)

# Load factors
lst_factors = LSTFactors(
    "ForLST/amenitykernel_cleaned.tif",
    "ForLST/building density.tif",
    "ForLST/CBD_cleaned (1).tif",
    # ... add all 11 factors
)

# Initialize simulator
simulator = LSTSimulator(lst_data, lst_factors, patch_size=64)
simulator.build_model()

# Train
history = simulator.train(epochs=100, batch_size=8)

# Predict current LST
predicted_lst = simulator.predict_lst(
    landuse_map=lst_data.arr_lu3,
    ndvi_map=lst_data.arr_ndvi3,
    emissivity_map=lst_data.arr_emiss3
)

# Evaluate
rmse, mae, r2 = simulator.evaluate(lst_data.arr_lst3, predicted_lst)

# Simulate future LST
future_predictions = simulator.simulate_future_lst(
    future_landuse_2035,
    future_landuse_2045
)
```

### Command Line Execution

```bash
python main.py
```

## 🏗️ Model Architecture

### Overview

```
Input (Land Use + NDVI + Emissivity + 11 Factors + 6 Derived Features)
         ↓
    Initial Conv (7×7, stride=2) + MaxPool
         ↓
    Enhanced ResNet Blocks with SE Attention (4 blocks)
         ↓
  Spatial Transformer 1 (8 heads, 3 layers)
         ↓
    Enhanced ResNet Blocks with SE (4 blocks)
         ↓
  Spatial Transformer 2 (8 heads, 2 layers)
         ↓
    Final ResNet Blocks with SE (2 blocks)
         ↓
   Thermal Regulation Module (Adaptive scaling)
         ↓
    Multi-Scale Feature Fusion
         ↓
    Upsampling + Skip Connections
         ↓
  Physical Thermal Constraints (UHI, NDVI cooling, smoothing)
         ↓
    LST Prediction (°C)
```

### Key Components

#### 1. **Enhanced Residual Blocks with Squeeze-and-Excitation**
- Channel-wise attention mechanism
- Adaptive feature recalibration
- Better capture of thermal patterns
- Improved gradient flow

#### 2. **Spatial Transformers**
- Memory-efficient multi-head attention (reduction ratio: 4)
- 2D positional encoding for spatial awareness
- 8 attention heads per layer
- Captures long-range thermal dependencies

#### 3. **Thermal Regulation Module**
- Global context aggregation
- Adaptive temperature scaling
- Learned thermal adjustment factor
- Prevents unrealistic temperature predictions

#### 4. **Multi-Scale Feature Fusion**
- Skip connections from initial features
- Progressive refinement
- Edge-preserving smoothing
- Spatial coherence enforcement

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Patch Size | 64×64 | Input patch dimensions |
| Batch Size | 8 | Training batch size |
| Initial Learning Rate | 0.0005 | Starting learning rate |
| Max Learning Rate | 0.005 | Peak learning rate (OneCycle) |
| Weight Decay | 1e-5 | L2 regularization |
| Epochs | 100 | Training epochs (no early stopping) |
| Gradient Clipping | 0.5 | Max gradient norm |
| Num Heads | 8 | Attention heads |
| Transformer Layers | 3 (first), 2 (second) | Transformer depth |
| Reduction Ratio | 4 | Spatial attention reduction |
| SE Reduction | 16 | SE block channel reduction |

## 📊 Input Features

### Basic Features (3)
1. **Land Use Classification** - Discrete classes (urban, vegetation, water, etc.)
2. **NDVI** - Vegetation index (-1 to 1)
3. **Surface Emissivity** - Thermal emissivity (0 to 1)

### Socioeconomic and Environmental Factors (11)
4. **Amenity Density** - Kernel density of amenities
5. **Building Density** - Building footprint density
6. **CBD Distance** - Distance to Central Business District
7. **Commercial Density** - Commercial area kernel density
8. **Industrial Density** - Industrial area kernel density
9. **Nighttime Lights (NTL)** - Light intensity
10. **Population Density** - People per unit area
11. **Restricted Areas** - Development constraints
12. **Road Network Density** - Road coverage
13. **Road Distance** - Distance to nearest road
14. **Terrain Slope** - Surface slope (0-90°)

### Derived Features (6)
15. **Urban Intensity** - Building density × Urban mask
16. **Vegetation Cooling** - NDVI × Vegetation mask
17. **CBD Inverse Distance** - 1/(CBD distance + ε)
18. **Road Inverse Distance** - 1/(Road distance + ε)
19. **Urban Heat** - Building density × Nighttime lights
20. **Elevation Effect** - exp(-slope/45°)

**Total: 20 input channels**

## 🌡️ Thermal Constraints

The model applies physics-based constraints to ensure realistic LST predictions:

### 1. **Urban Heat Island (UHI) Effect**
- Enhanced warming for urban areas (4.5°C - 8.5°C depending on expansion intensity)
- Considers building density, population, and nighttime lights
- Stronger effect for future urban expansion scenarios
- Spatial propagation of heat

### 2. **NDVI-Based Cooling**
- Vegetation provides cooling effect (-2.5°C per unit NDVI)
- Saturation limits prevent over-cooling
- Maximum cooling: -5°C

### 3. **Spatial Smoothing**
- Gaussian filtering (σ=0.8) for spatial coherence
- Edge-preserving blending
- Adaptive smoothing based on urban density
- More smoothing in dense urban areas (10-30%)

### 4. **Temperature Range Constraints**
- Minimum: Historical min - 3°C (≥15°C)
- Maximum: Historical max + 10°C (≤50°C)
- Prevents physically unrealistic values

### 5. **Temporal Warming Trends**
- Historical warming rate analysis
- Future projections with 50% acceleration
- Minimum 1.5°C warming for 10-year projection (default 1.8°C)
- Climate change consideration

### 6. **Water Body Effects**
- Water maintains cooler temperatures
- Special handling for water class (class 3)

## 📈 Evaluation Metrics

### Regression Metrics
- **RMSE (Root Mean Square Error)**: Overall prediction accuracy (°C)
- **MAE (Mean Absolute Error)**: Average absolute error (°C)
- **R² Score**: Explained variance (0 to 1, higher is better)
- **Bias**: Systematic over/under-prediction (°C)
- **MSE (Mean Square Error)**: Squared error metric

### Relative Metrics
- **Relative MAE**: MAE normalized by mean actual temperature
- **Relative RMSE**: RMSE normalized by mean actual temperature

### Skill Metrics
- **Skill Score**: 1 - (MSE_model / MSE_climatology)
  - Measures improvement over simple climatology
  - Values > 0 indicate skill
  - Values > 0.5 indicate good skill

### Urban Heat Island Metrics
- **UHI Intensity**: Temperature difference between urban and vegetation areas (°C)
- **LST by Land Use Class**: Mean temperature for each class
- **Urban-specific warming**: Temperature change in urban areas

## 📁 Output Files

### Model Checkpoints
- `best_lst_model.pth` - Best validation model with metadata
- `lst_checkpoint_epoch_X.pth` - Periodic checkpoints (every 25 epochs)

### LST Predictions
- `predicted_2025_lst_enhanced.tif` - Validation prediction
- `predicted_lst_2035_enhanced.tif` - Future LST for 2035
- `predicted_lst_2045_enhanced.tif` - Future LST for 2045

### Visualizations
- `lst_comparison_maps.png` - Historical vs predicted LST maps
- `lst_trends.png` - Temperature trend analysis
- `lst_enhanced_training_history.png` - Training metrics (loss, MAE, learning rate)

### File Formats

**GeoTIFF Predictions:**
- Data Type: Float32
- NoData Value: -9999
- Unit: Degrees Celsius (°C)
- Projection: Same as input
- Spatial resolution: Same as input

## 🔧 Advanced Configuration

### Memory Optimization

For limited GPU memory:

```python
# Reduce batch size
simulator.train(epochs=100, batch_size=4)

# Reduce patch size
simulator = LSTSimulator(lst_data, lst_factors, patch_size=32)

# Adjust reduction ratio in model
# In AdvancedLSTResNetTransformerModel:
self.transformer1 = SpatialTransformer(
    reduction_ratio=8  # Increase for more memory savings
)
```

### Custom Loss Weights

Adjust loss function weighting:

```python
# In train() method
loss = 0.3 * mse_loss + 0.4 * mae_loss + 0.3 * huber_loss
# Modify weights as needed (must sum to ~1.0)
```

### Thermal Constraint Tuning

Modify UHI enhancement:

```python
# In _apply_thermal_constraints()
uhi_enhancement = urban_expansion_intensity * 8.5  # Adjust multiplier
```

### Learning Rate Adjustment

```python
# In build_model()
self.optimizer = optim.AdamW(
    self.model.parameters(),
    lr=0.001,  # Increase for faster convergence
    weight_decay=1e-5
)
```

## 🐛 Troubleshooting

### Common Issues

**1. RMSE > 5°C**
- Check input data quality (NoData values, outliers)
- Verify LST data is in Celsius (not Kelvin)
- Increase training epochs
- Adjust loss function weights

**2. Predictions All Similar Values**
- Check feature normalization
- Verify sufficient data variance
- Increase model capacity
- Reduce regularization

**3. Unrealistic Temperature Predictions**
- Review thermal constraint parameters
- Check input feature ranges
- Verify land use classification quality
- Adjust UHI enhancement factor

**4. Training Loss Not Decreasing**
- Reduce learning rate
- Check for NaN values in data
- Increase batch size
- Verify gradient clipping

**5. CUDA Out of Memory**
- Reduce batch_size to 4 or 2
- Reduce patch_size to 32
- Increase reduction_ratio to 8
- Clear unused variables

**6. NaN in Predictions**
- Check for NaN in input data
- Verify normalization parameters
- Review thermal constraint limits
- Check temperature range constraints

## 📊 Performance Benchmarks

Typical performance on NVIDIA RTX 3090 (24GB):

| Task | Time | Memory |
|------|------|--------|
| Training (100 epochs) | ~60-90 min | 16-20 GB |
| Single LST prediction (2048×2048) | ~20 sec | 10 GB |
| Future scenario simulation (2 years) | ~45 sec | 12 GB |

**Expected Accuracy (well-calibrated model):**
- RMSE: 1.5-3.0°C
- MAE: 1.0-2.5°C
- R²: 0.75-0.90
- Skill Score: 0.5-0.8

## 📚 Dependencies

Core dependencies:
- PyTorch >= 2.0.0
- GDAL >= 3.0.0
- NumPy >= 1.21.0
- SciPy >= 1.8.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- pandas >= 1.4.0

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Update documentation
5. Submit a pull request


## 🗺️ Future Work

- [ ] Multi-temporal attention mechanisms
- [ ] Uncertainty quantification
- [ ] Integration with climate models
- [ ] Real-time LST monitoring
- [ ] Support for multiple satellite sensors
- [ ] Ensemble predictions
- [ ] Web-based visualization dashboard
- [ ] Pre-trained model weights

---

**Note**: This is research software. Results should be validated against ground truth measurements and local climate knowledge before use in operational applications.
