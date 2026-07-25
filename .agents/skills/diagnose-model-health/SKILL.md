---
name: diagnose-model-health
description: >-
  Runs a strict, predefined battery of diagnostic tests against a model checkpoint to verify network health, detect mode collapse, test tensor scaling, and validate Grad-CAM explainability outputs. Aggregates results into a single markdown report.
---

# Diagnose Model Health

## Overview
This skill executes a predefined gauntlet of diagnostic scripts against a `HierarchicalUNet` model checkpoint. It is specifically designed to detect catastrophic issues like Mode Collapse, activation saturation (tensor scaling mismatch), and dead gradients. 

It runs the following tests in order:
1. `test_weights.py`
2. `test_div255.py` / `test_bias.py`
3. `test_features.py` / `test_normal.py`
4. `test_cam.py`

## Dependencies
- None required to run the scripts, but assumes the project environment is set up.

## Quick Start
To diagnose a checkpoint, simply ask the agent:
> "Run the model health diagnostic on unet_hierarchical_best_oct5k.pth"

## Utility Scripts

### The Gauntlet Runner
The skill provides a master runner script that coordinates the tests and aggregates the output.

**Command:**
```bash
uv run .agents/skills/diagnose-model-health/scripts/runner.py \
  --checkpoint "/path/to/checkpoint.pth" \
  --output "diagnostics_report.md"
```

**Required Arguments:**
- `--checkpoint`: Absolute or relative path to the `.pth` weights you want to diagnose.
- `--output`: File path to save the final markdown report.

**Behavior:**
- The script executes the tests sequentially. 
- If a script throws an error, the runner logs the stack trace to the markdown file and forces the pipeline to continue to the next test.
- Once finished, the agent MUST read the `--output` markdown file using the `view_file` tool to interpret the results for the user.

## Common Mistakes
1. **Forgetting to read the output report**: The agent must read the generated markdown report to summarize the findings for the user.
2. **Missing checkpoint**: Ensure the `--checkpoint` provided actually exists.
