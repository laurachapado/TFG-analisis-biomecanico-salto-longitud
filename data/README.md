# Data required to reproduce the project

Raw competition material is intentionally not included in this public repository.

The end-to-end workflow expects the following source material:

1. **Competition videos** (`.avi` / `.mp4`) recorded from a lateral view at 240 fps.
2. **Official competition results** used to associate each video with validity and jump distance.
3. **Historical athlete results** used to estimate each athlete's previous performance level.

The video-extraction stage generates intermediate files such as:

- `eventos_automaticos_<video>.csv`
- `keypoints_angulos_<video>.csv`
- `variables_velocidad_batida_<video>.csv`
- calibration and diagnostic files

The analysis notebook subsequently builds the cleaned attempt-level datasets used for modelling.

## Study material

The final thesis study uses real competition material from the 2025 Spanish Indoor Championships:

- 66 attempts
- 14 athletes
- 42 valid attempts
- 24 foul attempts
- 240 fps lateral video

Because the original footage and athlete-linked source files are not distributed here, the repository documents the complete research workflow but is not fully executable from a fresh clone without those inputs.
