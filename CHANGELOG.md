# Changelog

## Recent Changes (2026-01-27)

### Major Features

#### 1. Unified Simulator Interface
- **Command-line simulator selection**: Added required `--simulator` flag to both `qps_search.py` and `parallel_search.py`
- **No more file editing**: Switch between BLIS and Vidur with a single command-line argument
- **Consistent API**: Both simulators use identical function signatures
- **Examples updated**: All documentation now shows explicit `--simulator blis` or `--simulator vidur`

```bash
# Before: Had to edit USE_VIDUR = True/False in source files
# After: Just pass --simulator flag
python qps_search.py --config test.json --simulator blis
python parallel_search.py --configs configs.yaml --simulator vidur
```

#### 2. Runtime Tracking
- **Total runtime reporting**: All tools now report total search time
- **Per-config runtime**: `parallel_search.py` reports runtime for each configuration
- **JSON output enhanced**: Runtime data included in saved results
- **Console display**: Real-time runtime display during execution

```
Results:
  Max QPS: 15.50
  Total Runtime: 182.47 seconds

[Config 1] ✅ Complete - Max QPS: 15.50 (Runtime: 45.32s)
```

#### 3. YAML Support in qps_search.py
- **Flexible config format**: qps_search.py now supports both JSON and YAML files
- **Grid search handling**: Automatically uses first value from parameter lists
- **Explicit configs**: Uses first config from explicit config lists
- **Unified workflow**: Same config files work across all tools

```bash
# Now works with YAML
python qps_search.py --config examples/configs_grid_search.yaml --simulator blis
```

#### 4. Roofline Model Visibility
- **Configuration display**: Shows when roofline model is enabled
- **Parameter visibility**: Displays `hardware_config` and `model_config_folder_base` in output
- **Status indicators**: Clear "Roofline Model: ENABLED/DISABLED" messages

```
Configuration:
  Hardware Config: hardware_config.json
  Model Config Base: model_configs
  Roofline Model: ENABLED
```

### Bug Fixes

#### 1. Fixed Multiprocessing Simulator Selection
- **Issue**: `parallel_search.py` always used BLIS even when `--simulator vidur` was specified
- **Root cause**: Global `USE_VIDUR` variable didn't propagate to worker processes
- **Fix**: Pass simulator choice as function parameter to each worker
- **Impact**: Vidur now works correctly in parallel search

#### 2. Fixed JSON Serialization with NumPy 2.0
- **Issue**: `AttributeError: np.float_ was removed in NumPy 2.0`
- **Root cause**: NumPy 2.0 removed deprecated type aliases like `np.float_` and `np.int_`
- **Fix**: Updated `convert_to_json_serializable()` to use abstract base classes (`np.integer`, `np.floating`)
- **Impact**: Works with both NumPy 1.x and 2.x

```python
# Before (NumPy 1.x only):
elif isinstance(obj, (np.float_, np.float16, ...)):
    return float(obj)

# After (NumPy 1.x and 2.x):
elif isinstance(obj, (float, np.floating)):
    return float(obj)
```

#### 3. Fixed Vidur Temp Directory Collisions
- **Issue**: Parallel Vidur runs created directories with same timestamp
- **Root cause**: Timestamp-only naming in concurrent processes
- **Fix**: Added process ID to temp directory name
- **Impact**: No more race conditions in parallel execution

```python
# Before:
temp_dir = f'vidur_sim_{timestamp}'

# After:
temp_dir = f'vidur_sim_{timestamp}_{pid}'
```

#### 4. Added Simulation Failure Handling
- **Issue**: Invalid metrics (None, zeros, NaN) caused crashes or incorrect SLO checks
- **Root cause**: No validation of simulation output
- **Fix**: Added comprehensive validation in both `blis_runner.py` and `vidur_runner.py`
- **Impact**: Graceful handling of failed simulations

### Code Cleanup

#### 1. Removed Redundant Code from vidur_runner.py
- **Removed**: `display_metrics_category()`, `violates_slo()`, `find_max_qps()`, `main()`
- **Reason**: These functions are now in `qps_search.py` and imported by all tools
- **Impact**: 726 lines → 383 lines (47% reduction)
- **Benefits**: Single source of truth, easier maintenance

#### 2. Consolidated Imports
- **Change**: All tools now import both simulators upfront
- **Benefit**: Cleaner code, no conditional imports
- **Example**:
  ```python
  from vidur_runner import run_vidur
  from blis_runner import run_blis
  ```

#### 3. Removed Test File
- **Deleted**: `test_vidur_runner.py` (tested non-existent functions)
- **Reason**: Functions it tested (`convert_vllm_config_to_vidur`, `generate_vidur_yaml`) were never implemented
- **Alternative**: Integration testing through `qps_search.py` and `parallel_search.py`

### Documentation Updates

#### 1. New Main README.md
- **Comprehensive overview** of all tools
- **Quick start guide** with examples
- **Configuration file formats** (JSON and YAML)
- **Feature comparison** (BLIS vs Vidur)
- **Output examples** (console and JSON)

#### 2. Updated SETUP.md
- **Simulator flag requirement** in all examples
- **New features section** (runtime tracking, YAML support)
- **Updated commands** with `--simulator` flag

#### 3. Updated Config Files
- **test_config.json**: Added `model_config_folder_base` and `hardware_config` for roofline model (serves as both standard and roofline example)
- **test_config_small.json**: Quick testing config with fewer requests for faster testing

### Performance Improvements

- **Parallel execution**: Worker processes now correctly use specified simulator
- **Efficient temp management**: No directory collisions in Vidur parallel runs
- **Early failure detection**: Invalid metrics detected immediately
- **Caching**: Results caching prevents duplicate simulations

### Breaking Changes

#### Required `--simulator` Flag
**Before**:
```bash
python qps_search.py --config test.json
python parallel_search.py --configs configs.yaml
```

**After**:
```bash
python qps_search.py --config test.json --simulator blis
python parallel_search.py --configs configs.yaml --simulator blis
```

**Reason**: Explicit simulator choice prevents accidental use of wrong simulator.

### Compatibility

- **Python**: 3.11+ required (unchanged)
- **NumPy**: Compatible with both NumPy 1.x and 2.x
- **BLIS**: Uses `openevolve` branch (unchanged)
- **Config files**: Backward compatible (BLIS-specific fields ignored by Vidur)

### Testing

All changes have been validated with:
- ✅ Syntax validation (`python -m py_compile`)
- ✅ BLIS simulator runs
- ✅ Vidur simulator runs (when available)
- ✅ Parallel execution with both simulators
- ✅ JSON output generation
- ✅ YAML config parsing

### Migration Guide

If you have existing scripts:

1. **Add `--simulator` flag** to all commands:
   ```bash
   # Old:
   python qps_search.py --config test.json

   # New:
   python qps_search.py --config test.json --simulator blis
   ```

2. **Update config files** to include roofline parameters (optional):
   ```json
   {
     "model_config_folder_base": "model_configs",
     "hardware_config": "hardware_config.json"
   }
   ```

3. **Use YAML configs** for easier grid search:
   ```yaml
   tp: [1, 2]
   batch_size: [128, 256, 512]
   ```

### Known Limitations

- **Vidur setup**: Still requires GPU profiling for new models
- **BLIS roofline**: Requires model config files for unprofiled models
- **Single-node only**: Multi-replica not yet supported in Vidur wrapper

### Future Enhancements

- [ ] Add `--output` flag to `qps_search.py`
- [ ] Support for custom metrics in SLOs
- [ ] Automatic model detection from HuggingFace
- [ ] Config validation before simulation
- [ ] Resume from checkpoint for long searches
- [ ] Interactive progress visualization

## Contributors

- Implementation: Claude (Anthropic)
- Supervision: Dipanwita Guhathakurta

---

**Note**: This changelog covers changes made during the integration and cleanup phase. See individual STEP*_SUMMARY.md files for earlier development history.
