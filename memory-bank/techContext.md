# VAME Technical Context

## Development Environment

### Core Requirements
- Python 3.11 or higher
- CUDA-capable GPU (recommended)
- Anaconda/Miniconda (recommended for environment management)

### Operating System Support
- Linux (primary development platform)
- Windows (supported)
- macOS (supported)

## Technology Stack

### Primary Technologies
1. **Python**
   - Primary development language
   - Version 3.11+ required
   - Black formatting enforced

2. **PyTorch**
   - Deep learning framework
   - Version 2.2.2 tested
   - CUDA support for GPU acceleration

3. **NumPy/SciPy**
   - Numerical computations
   - Data manipulation
   - Scientific computing utilities

4. **xarray**
   - Multi-dimensional labeled arrays
   - NetCDF file handling
   - Data structure management

### Supporting Libraries
1. **Data Processing**
   - pandas: Data manipulation
   - h5py: HDF5 file handling
   - netCDF4: NetCDF file support

2. **Visualization**
   - matplotlib: Basic plotting
   - seaborn: Statistical visualization
   - UMAP: Dimensionality reduction

3. **Machine Learning**
   - scikit-learn: ML utilities
   - tqdm: Progress tracking
   - torch.utils.data: Data loading

4. **Development Tools**
   - pytest: Testing framework
   - black: Code formatting
   - docutils: Documentation

## Dependencies

### Core Dependencies
```yaml
- python>=3.11
- pytorch>=2.2.2
- numpy
- scipy
- xarray
- pandas
- h5py
- netcdf4
- matplotlib
- seaborn
- scikit-learn
- tqdm
```

### Optional Dependencies
```yaml
- jupyter  # For notebook examples
- sphinx   # For documentation
- pytest   # For testing
- black    # For code formatting
```

## Technical Constraints

### Hardware Requirements
1. **Minimum**
   - CPU: Multi-core processor
   - RAM: 8GB
   - Storage: 10GB free space

2. **Recommended**
   - GPU: NVIDIA GTX 1080 Ti or better
   - RAM: 16GB+
   - Storage: 50GB+ free space

### Performance Considerations
1. **GPU Computing**
   - Required for efficient training
   - CUDA support essential
   - Memory usage scales with data size

2. **Data Processing**
   - Large dataset handling
   - Memory-efficient operations
   - Batch processing capabilities

3. **Storage**
   - NetCDF file format for data
   - Efficient data compression
   - Modular data organization

## Development Setup

### Environment Setup
```bash
# Create conda environment
conda env create -f VAME.yaml

# Activate environment
conda activate vame

# Install VAME
python setup.py install

# Install PyTorch with CUDA
# (version depends on system configuration)
```

### Project Structure
```
vame/
├── src/
│   └── vame/
│       ├── analysis/
│       ├── initialize_project/
│       ├── io/
│       ├── model/
│       ├── preprocessing/
│       └── visualization/
├── tests/
├── docs/
├── examples/
└── config files
```

## Integration Points

### Input Formats
1. **Pose Estimation Data**
   - DeepLabCut CSV format
   - SLEAP HDF5 format
   - LightningPose format
   - NWB format support

### Output Formats
1. **Results**
   - NetCDF files
   - JSON state files
   - MP4 video outputs
   - Analysis reports

### External Tools
1. **Pose Estimation**
   - DeepLabCut
   - SLEAP
   - LightningPose

2. **Data Format**
   - NWB (Neurodata Without Borders)
   - Movement package

## Security and Access

### Data Security
- Local data processing
- No cloud dependencies
- File-based storage
- User-managed data

### Access Control
- File system based
- No authentication required
- Local execution only
- User permissions based

## Monitoring and Logging

### Logging System
- Python logging framework
- Progress tracking with tqdm
- Error reporting
- State management

### Performance Monitoring
- Model training metrics
- Loss tracking
- GPU utilization
- Memory usage

## Backup and Recovery

### State Management
- JSON state tracking
- Checkpoint saving
- Model snapshots
- Progress recovery

### Data Protection
- Original data preservation
- Intermediate results saving
- Model checkpoint system
- Configuration backups
