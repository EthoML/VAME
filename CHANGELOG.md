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