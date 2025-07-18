# v0.11.0

### Fixes

- Loose torch version `torch>=2.3.1`
- Updated `pynwb==3.0.0`
- Updated `movement==0.6.1`
- Updated `ndx-pose==0.2.2`
- Updated `ndx-vame==0.2.2`
- Remove deprecated `verbose` argument for `ReduceLROnPlateau`
- Fix gpu usage ([Issue #173](https://github.com/EthoML/VAME/issues/173))
- Fix redundant inference when estimating motifs ([Issue #176](https://github.com/EthoML/VAME/issues/176))
- Fix UMAP, now it uses all sessions (cohort) ([Issue #177](https://github.com/EthoML/VAME/issues/177))
- Fix preprocessing ([Issue #180](https://github.com/EthoML/VAME/issues/180))


### Features

- Export Project results to NWB files ([Issue #171](https://github.com/EthoML/VAME/issues/171))
- UMAP figures with proper labels and colors
- Interactive UMAP figures with Plotly
- tensorboard tracking during model training
- Batch processing speed up latent space embeddings
- Project-wide random state with `project_random_state` in config
- Creates `pose_estimation_inspection` function for quick inspection of useful information


# v0.10.0

### Fixes

- Environment file for fixing GPU not found on Windows

### Features

- Better organization for Report and UMAP results
- Logs True by default for all functions


# v0.9.0

### Fixes

- Fix movement sets fps to None, which is not accepted by ds.to_netcdf ([Issue #148](https://github.com/EthoML/VAME/issues/148))
- Re-order dataset dimensions to be in line with movement
- Fix imports for download sample data
- Fix get_version ([Issue #164](https://github.com/EthoML/VAME/issues/164))
- Removed smoothing filter on community labels and save community fig as a .pdf
- Save pose segmentation data after each session instead of segmentation algorithm method ([Issue #128](https://github.com/EthoML/VAME/issues/128))
- Fix IQR filter ([Issue #158](https://github.com/EthoML/VAME/issues/158))
- Fix bug with motif usage counts in community function ([PR #156](https://github.com/EthoML/VAME/pull/156))
- Removed redundant folder in the results folder ([PR #156](https://github.com/EthoML/VAME/pull/156))

### Features

- Update torch version to 2.5.1
- Update `movement==0.5.0`
- Update netCDF engine to `netCDF4`
- Add VAME package version to config ([Issue #147](https://github.com/EthoML/VAME/issues/147))
- Improve test/train set split ([Issue #146](https://github.com/EthoML/VAME/issues/146))
- re-scaling as an extra preprocessing step, optional with default value False
- visualization of the raw, threshold-cleaned, egoaligned, and IQR-cleaned (non-z-scored) time series example and X-vs-Y scatter plots that show how the data is being transformed
- Points cloud scatter plot
- Add tests for M1 ([Issue #149](https://github.com/EthoML/VAME/issues/149))
- General improvements to save_state in some functions
- Community analysis, motif videos and community videos are now done automatically for all segmentation algorithms


# v0.8.0

### Fixes

- OS aware dependencies installation, specially important for pytorch

### Features

- Initialize project and run pipeline without videos ([Issue #129](https://github.com/EthoML/VAME/issues/129))
- Allow adding videos later on in the project ([Issue #129](https://github.com/EthoML/VAME/issues/129))
- Run on Google colab tags in documentation pages  ([Issue #144](https://github.com/EthoML/VAME/issues/144))


# v0.7.0

### Fixes

- Egocentric alignment outputs with incorrect column order  ([Issue #96](https://github.com/EthoML/VAME/issues/96))
- Slow align egocentrical data ([Issue #113](https://github.com/EthoML/VAME/issues/113))
- Standardized config argument across all functions

### Features

- User friendly way to select reference points for alignment ([Issue #89](https://github.com/EthoML/VAME/issues/89))
- Adopt movement Xarray data format ([Issue #111](https://github.com/EthoML/VAME/issues/111))
- Relocate IQR cleaning into preprocessing ([Issue #22](https://github.com/EthoML/VAME/issues/22))
- Created preprocessing module ([Issue #119](https://github.com/EthoML/VAME/issues/119))
- Separate module for visualization of results ([Issue #127](https://github.com/EthoML/VAME/issues/127))
- Further improvements to Pipeline


# v0.6.0

### Fixes

- Removed auto adding date from poject names (#105)
- Renamed `parametrization` to `segmentation_algorithm` (#106)
- Renamed `file` to `session` (#107)
- Removed old doctrings at top of files (#108)
- Build compatibility with SLEAP (#39)
- Build compatibility with LightningPose (#115)
- `num_features` should be extracted from DLC data (#100)
- Create VAME Project Class (#42)

### Features
- Improve handling of user input (cli) when `all_data==no`
- `vame.report()` saves figures to `<project_path>/reports/` folder
- Created `VAMEPipeline` class that makes it straightforward to run an entire pipeline, and also to pick it up from where it last stopped
- Use https://github.com/neuroinformatics-unit/movement as main data ingestion interface
