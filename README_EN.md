# Spatio-Temporal Traffic Data Preprocessing Tool

*阅读其他语言版本：[中文](README.md)*

This project provides spatio-temporal traffic data preprocessing and graph network construction tools, supporting various traffic datasets.

## Acknowledgments

Thanks to the [FlashST](https://github.com/icecity96/FlashST-master) project for providing partial datasets and graph processing methods.

## Project Structure

```
.
├── CA_District5/                   # California District 5 traffic data
├── NYC_BIKE/                      # New York City bike sharing data
├── PEMS03/                        # PEMS03 traffic flow data
├── PEMS07M/                       # PEMS07M traffic speed data
├── chengdu_didi/                  # Chengdu DiDi ride-hailing data
├── data_prepare/                  # Data preprocessing module
│   ├── data_conf/                 # Configuration files directory
│   │   ├── CA_District5_3dim_construct_12.conf
│   │   ├── METRLA_3dim_construct_12.conf
│   │   ├── NYC_BIKE_3dim_construct_12.conf
│   │   ├── PEMS03_3dim_construct_12.conf
│   │   ├── PEMS07M_3dim_construct_12.conf
│   │   ├── PEMS08_3dim_construct_12.conf
│   │   └── chengdu_didi_3dim_construct_12.conf
│   └── prepareData_STAEformer.py  # Main data preprocessing script
└── graph_process.py               # Graph network processing utilities
```

## Main Features

### 1. Data Preprocessing (`data_prepare/prepareData_STAEformer.py`)

This module is used to construct datasets required for model training, with main features including:

- **Multi-dataset Support**: Supports various traffic datasets including PEMS03/04/07/08, NYC_BIKE, CA_District5, chengdu_didi, etc.
- **Temporal Feature Embedding**: Automatically adds temporal dimension features such as day and week
- **Data Normalization**: Applies Min-Max normalization to temporal features
- **Data Splitting**: Automatically splits data into training, validation, and test sets according to configuration files
- **Sequence Construction**: Constructs time series samples based on historical and prediction time steps

### 2. Graph Network Processing (`graph_process.py`)

This module provides rich traffic network graph processing methods:

#### Graph Construction and Loading
- Multiple data format support (CSV, NPY, PKL)
- Adjacency matrix and distance matrix construction
- Weight matrix calculation

#### Graph Normalization Methods
- **Symmetric Normalization**: `get_normalized_adj()` - Symmetric normalized adjacency matrix
- **Asymmetric Normalization**: `asym_adj()` - Row normalization processing
- **Message Passing Normalization**: `symmetric_message_passing_adj()` - Symmetric message passing

#### Laplacian Matrix Computation
- **Standard Laplacian**: `calculate_normalized_laplacian()`
- **Symmetric Normalized Laplacian**: `calculate_symmetric_normalized_laplacian()`
- **Scaled Laplacian**: `calculate_scaled_laplacian()`
- **Laplacian Positional Encoding**: `cal_lape()` - Positional encoding for graph neural networks

#### Transition Matrix
- **Random Walk Matrix**: `transition_matrix()` - For random walks on graphs
- **Bidirectional Transition Matrix**: Supports bidirectional random walks

## Usage

### Data Preprocessing

1. **Configure Dataset Parameters**:
   ```bash
   # Edit the corresponding configuration file, for example:
   data_prepare/data_conf/PEMS08_3dim_construct_12.conf
   ```

   Configuration file example:
   ```ini
   [Data]
   num_of_vertices  = 170        # Number of nodes
   time_slice_size = 5           # Time slice size (minutes)
   train_ratio = 0.6             # Training set ratio
   val_ratio = 0.2               # Validation set ratio
   test_ratio = 0.2              # Test set ratio
   data_file = ../data/PEMS08/data.npz  # Data file path
   output_dir = ../data/PEMS08   # Output directory
   
   [Training]
   num_his = 12                  # Historical time steps
   num_pred = 12                 # Prediction time steps
   ```

2. **Run Data Preprocessing**:
   ```bash
   cd data_prepare
   python prepareData_STAEformer.py --config ./data_conf/PEMS08_3dim_construct_12.conf
   ```

3. **Output Results**:
   - Generates `.npz` file containing training, validation, and test sets
   - Data format: `(num_samples, time_steps, num_nodes, feature_dims)`
   - Includes original features + temporal features (day, week)

### Graph Network Processing

Import and use graph processing functions in your code:

```python
from graph_process import *

# Load and preprocess graph data
args.dataset_graph = 'PEMS08'  # Set dataset name
pre_graph_dict(args)           # Preprocess graph data

# Use processed graph data
adj_matrix = args.A_dict['PEMS08']        # Normalized adjacency matrix
laplacian_pe = args.lpls_dict['PEMS08']   # Laplacian positional encoding
```

## Supported Datasets

| Dataset | Type | Nodes | Time Interval | Data Dimension |
|---------|------|-------|---------------|----------------|
| PEMS03 | Traffic Flow | 358 | 5 min | Flow |
| PEMS04 | Traffic Flow | 307 | 5 min | Flow |
| PEMS07 | Traffic Flow | 883 | 5 min | Flow |
| PEMS08 | Traffic Flow | 170 | 5 min | Flow |
| PEMS07M | Traffic Speed | - | 5 min | Speed |
| NYC_BIKE | Bike Demand | - | 30 min | Demand |
| CA_District5 | Traffic Flow | - | 5 min | Flow |
| chengdu_didi | Traffic Index | - | 10 min | Index |

## Key Features

- ✅ Support for multiple traffic dataset formats
- ✅ Automatic temporal feature engineering
- ✅ Flexible data splitting configuration
- ✅ Rich graph processing algorithms
- ✅ Laplacian positional encoding support
- ✅ Multiple graph normalization methods
- ✅ Easy-to-extend modular design

## Dependencies

```
numpy
pandas
scipy
torch
configparser
```

## License

Parts of this project's code and datasets are derived from the FlashST project. Please follow the corresponding open source license.