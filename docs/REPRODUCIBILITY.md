# Reproducibility and file map

## Execution order

```text
raw competition video
        |
        v
src/extract_video_features.py
        |
        +--> keypoints / angles
        +--> detected events / take-off
        +--> velocity-related variables
        |
        v
src/align_post_takeoff_frames.py
        |
        v
common post-take-off reference frame
        |
        v
notebooks/analysis_and_modeling.ipynb
        |
        +--> feature-block integration
        +--> quality control / cleaning
        +--> official results merge
        +--> athlete reference level
        +--> classification / regression
```

## What has been validated in this repository

- Both cleaned Python scripts pass Python syntax compilation.
- Every code cell in the cleaned notebook parses successfully as Python.
- The notebook structure has been reorganised and all stored execution outputs have been removed.
- The original thesis PDF and selected defence visuals are included for methodological context.

## What cannot be validated without the private source material

A complete end-to-end execution requires the original competition videos and source/result files referenced above. Those files were not part of the material available when this public repository was prepared, so full runtime reproduction has not been claimed.
