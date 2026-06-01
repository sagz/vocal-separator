# Multi-Hardware Lab & Gold Standard Library

This directory contains the reference audio inputs, testing scripts, and the "Gold Standard" library used for the Automated Multi-Hardware Lab.

## Overview
Current development utilizes an automated CI pipeline that tests code changes on multiple hardware backends, primarily:
- NVIDIA (CUDA) via a Windows self-hosted runner.
- Apple Silicon (MPS) via a macOS self-hosted runner.

These tests run headless and completely detached from the GUI. They invoke core algorithms on reference tracks and compare the output bit-by-bit against the Gold Standard library. 100% bit-exact parity is required for tests to pass.

## Adding New Reference Test Cases

Maintainers can expand the "Gold Standard" library by following these steps:

1. **Add Reference Audio**: Place your new `.wav` source file in the `tests/reference_audio/` directory.
2. **Generate Gold Standard Outputs**: Run the core algorithm on your new file using a known good, stable version of the code (e.g., from the `main` branch).
   ```bash
   python cli.py --audio tests/reference_audio/YOUR_TRACK.wav \
                 --export tests/gold_standard_library/mdx_net/ \
                 --model models/MDX_Net_Models/YOUR_MODEL.onnx \
                 --method "MDX-Net"
   ```
3. **Verify Bit-Exact Output**: Ensure the generated files in `tests/gold_standard_library/` are correct and sound as expected.
4. **Update CI Workflow**: If this is a new model or new specific test path, add a corresponding step in `.github/workflows/multi_hardware_lab.yml` to ensure it is tested automatically on future pull requests.
5. **Commit**: Commit the input audio and the resulting gold standard output files to the repository.
