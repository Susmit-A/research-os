# Implementation Guide Summary

**Created**: 2025-10-22
**Purpose**: Quick reference for resuming work on Tasks 4-8

---

## What Was Created

A comprehensive 900+ line implementation guide (`IMPLEMENTATION_GUIDE.md`) that provides everything needed to complete the remaining tasks (4-8) of the entropy regularization experiment.

---

## Key Contents

### 1. Architecture Documentation
- Complete system architecture diagram
- Data flow visualization
- Component integration overview
- File structure roadmap

### 2. Code Templates

**Task 4 - Baseline Training**:
- `src/training/trainer.py` - Complete training framework class (400+ lines)
- `src/training/train_baseline.py` - Training script for baseline model
- `tests/test_trainer.py` - Comprehensive test suite template

**Task 5 - Entropy-Regularized Training**:
- `src/training/train_entropy_reg.py` - Training script with entropy regularization
- Additional tests for entropy integration

**Task 7 - Evaluation Metrics**:
- `src/evaluation/metrics.py` - Information Gain and bias entropy measurement
- `src/evaluation/evaluate_model.py` - Model evaluation script
- `tests/test_metrics.py` - Metrics test suite

**Task 8 - Final Report**:
- `src/evaluation/generate_report.py` - Automated go/no-go report generation

### 3. Integration Points

Clear documentation of how to use existing components:
- Data loaders from Tasks 2 (`create_mit1003_dataloaders()`)
- Entropy regularizer from Task 3 (`EntropyRegularizer`)
- DeepGaze 3 model from Task 1 (`DeepGazeIII`)
- Configuration files (`baseline_config.yaml`, `entropy_reg_config.yaml`)

### 4. Testing Strategy

- TDD approach with tests written first
- Test organization and coverage requirements
- Instructions for using `engineer:test-runner` subagent
- Expected test counts: ~90+ total tests when complete

### 5. Implementation Checklists

Detailed checklists for each task:
- **Task 4**: 11 checkboxes (baseline training pipeline)
- **Task 5**: 8 checkboxes (entropy-regularized pipeline)
- **Task 6**: 6 checkboxes (parallel training execution)
- **Task 7**: 8 checkboxes (evaluation metrics)
- **Task 8**: 11 checkboxes (final evaluation and report)

### 6. Operational Instructions

**Task 6 - Training Execution**:
- Dataset download instructions
- SLURM job submission commands
- Training monitoring procedures
- Expected timeline (6-12 hours)
- Checkpoint verification

**Task 8 - Final Evaluation**:
- Evaluation script execution
- Report generation
- Go/no-go decision criteria
- Success metrics thresholds

### 7. Troubleshooting Guide

Common issues and solutions:
- Out of memory errors
- NaN losses
- Distributed training hangs
- Poor convergence

---

## How to Use This Guide

### For Immediate Next Steps

1. **Read** `IMPLEMENTATION_GUIDE.md` in full to understand architecture
2. **Start with Task 4**: Follow the checklist step-by-step
3. **Use TDD approach**: Write tests first, then implement
4. **Use test-runner subagent** in Step 6 as specified
5. **Mark tasks complete** in `tasks.md` as you go

### For Code Implementation

All templates include:
- **TODO comments** marking what needs to be implemented
- **Example code** showing the pattern to follow
- **Integration points** clearly marked
- **Type hints** for all function signatures
- **Docstrings** explaining purpose and arguments

### For Testing

Follow this pattern for each task:
1. Write all tests first (red phase)
2. Run tests to verify they fail
3. Implement code (green phase)
4. Use test-runner subagent to verify
5. Refactor if needed

---

## Expected Timeline

- **Task 4** (Baseline Training): 1-2 days implementation
- **Task 5** (Entropy Training): 0.5-1 day implementation
- **Task 6** (Training Execution): 8-12 hours compute time
- **Task 7** (Evaluation Metrics): 1 day implementation
- **Task 8** (Final Report): 0.5 day

**Total**: 3-5 days to completion

---

## Success Criteria

The final report will evaluate:

1. **Primary**: Bias entropy increase ≥ 5%
2. **Secondary**: In-domain IG degradation ≤ 2%
3. **Secondary**: OOD IG improvement > 0%
4. **Secondary**: Training stability confirmed

**Decision**: GO (proceed to full 11-week project) or NO-GO (abandon or revise approach)

---

## Current Status

- **Completed**: Tasks 1-3 (37.5%)
- **Remaining**: Tasks 4-8 (62.5%)
- **Tests Passing**: 64/64 (100%)
- **Implementation Guide**: ✅ Complete and ready

---

## Key Files to Reference

When implementing, frequently reference:

1. **IMPLEMENTATION_GUIDE.md** - Primary reference (this file)
2. **PROGRESS_SUMMARY.md** - Current status and completed work
3. **spec.md** - Original requirements and success criteria
4. **tasks.md** - Task breakdown and checklist
5. **TASK_1_SUMMARY.md**, **TASK_2_SUMMARY.md**, **TASK_3_SUMMARY.md** - Completed task documentation

---

## Integration with Existing Code

All templates are designed to integrate seamlessly with:

- ✅ Data loaders (Tasks 2) - `src/data/mit1003_loader.py`, `src/data/cat2000_loader.py`
- ✅ Entropy regularizer (Task 3) - `src/models/entropy_regularizer.py`
- ✅ DeepGaze 3 model (Task 1) - `src/models/deepgaze3.py`
- ✅ Configuration files (Task 1) - `configs/*.yaml`
- ✅ SLURM scripts (Task 1) - `scripts/run_*.sh`

No modification of existing code required - just implement the templates!

---

## Next Action

**When ready to resume implementation**:

```bash
# 1. Review the guide
cat IMPLEMENTATION_GUIDE.md

# 2. Start Task 4
# Follow Task 4 Implementation Checklist in the guide

# 3. Create test file
# Copy template from guide to tests/test_trainer.py

# 4. Create trainer module
# Copy template from guide to src/training/trainer.py

# 5. Follow TDD approach
# Write tests → Run tests → Implement → Verify → Repeat
```

---

*This summary provides a quick orientation to the comprehensive implementation guide. For full details, see IMPLEMENTATION_GUIDE.md*
