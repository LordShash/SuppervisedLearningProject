# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed

- **Visualizations in Result Tabs**: Fixed issue where visualizations in the "Confusion Matrix", "ROC Curve", and "Model Comparison" tabs remained empty despite successfully loaded metrics.
  - Fixed layout handling to properly delete old layouts instead of orphaning them
  - Added minimum sizes for visualization widgets to ensure they're visible
  - Added error handling to display error messages when exceptions occur
  - Added diagnostic code to check if PyQt5.QtChart is installed correctly
  - Added comprehensive logging to track visualization updates

### Added

- **Integration Tests**: Added tests to verify that visualizations are displayed correctly
  - Test for Confusion Matrix tab
  - Test for ROC Curve tab
  - Test for Model Comparison tab
  - Tests use dummy training results to simulate real data

### Changed

- **Improved Error Handling**: Enhanced error handling in visualization methods to provide better feedback to users
- **Better Logging**: Added detailed logging to help diagnose visualization issues
- **UI Improvements**: Added hint labels for missing data and improved widget sizing