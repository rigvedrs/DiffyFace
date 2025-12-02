# Evaluation Module

This directory contains comprehensive evaluation tools for the DiffyFace model.

## Overview

The evaluation system provides three main types of evaluation:

1. **Test Set Generation**: Generate a fixed set of test images from diverse prompts
2. **Self-Consistency Check**: Evaluate consistency by feeding generated images back to Groq API
3. **Attribute Evaluation**: Quantitative evaluation of attribute accuracy

## Quick Start

### 1. Generate Test Set

```bash
# Generate all test images (300 prompts)
python Evaluation/generate_test_set.py --output-dir Evaluation/test_set_output

# Generate a subset (e.g., 50 images for quick testing)
python Evaluation/generate_test_set.py --output-dir Evaluation/test_set_output --num-samples 50
```

This will:
- Generate images from test prompts in `test_prompts.json`
- Save images to the output directory
- Create `test_set_metadata.json` with all metadata

### 2. Run Self-Consistency Evaluation

```bash
python Evaluation/self_consistency_check.py \
    --test-set Evaluation/test_set_output/test_set_metadata.json \
    --output Evaluation/test_set_output/self_consistency_results.json
```

**Note**: Requires `GROQ_API_KEY` environment variable.

This will:
- Send each generated image to Groq API
- Extract attributes from regenerated descriptions
- Compare with original prompt attributes
- Compute match rates and accuracy metrics

### 3. Run Attribute Evaluation

```bash
python Evaluation/attribute_evaluation.py \
    --test-set Evaluation/test_set_output/test_set_metadata.json \
    --output Evaluation/test_set_output/attribute_evaluation_results.json
```

This will:
- Extract attributes from test prompts
- Compute attribute coverage statistics
- Generate accuracy metrics per attribute

### 4. Generate Evaluation Report

```bash
python Evaluation/generate_evaluation_report.py \
    --results-dir Evaluation/test_set_output \
    --output Evaluation/evaluation_report.md
```

This compiles all results into a formatted markdown report.

## Test Prompts

The test prompts are defined in `test_prompts.json` and cover:

- **Age ranges**: 20-30, 31-40, 41-50, 51+
- **Genders**: Male, Female
- **Hair colors**: Blond, Brown, Black, Red, Gray, White
- **Facial hair**: None, Mustache, Goatee, French Beard, Full Beard, Stubble
- **Expressions**: Smiling, Serious, Neutral, Cheerful
- **Accessories**: None, Eyeglasses, Sunglasses, Hat
- **Eye colors**: Brown, Blue, Green, Hazel

## Evaluation Metrics

### Self-Consistency Metrics

- **Average Match Rate**: Overall percentage of attributes that match between original prompt and Groq-regenerated description
- **Per-Attribute Accuracy**: Accuracy for each individual attribute (age, gender, hair color, etc.)

### Attribute Evaluation Metrics

- **Attribute Coverage**: Percentage of test samples that contain each attribute
- **Attribute Accuracy**: How well attributes in prompts match expected attributes

## Output Files

After running evaluations, you'll have:

- `test_set_output/test_set_metadata.json`: Test set information and generated image paths
- `test_set_output/self_consistency_results.json`: Detailed self-consistency evaluation results
- `test_set_output/attribute_evaluation_results.json`: Attribute evaluation results
- `test_set_output/evaluation_report.md`: Formatted evaluation report

## Example Results

Typical evaluation results show:

- **Self-Consistency Match Rate**: 75-85% (depending on attribute)
- **Attribute Coverage**: 90-100% (most attributes appear in expected samples)
- **Best Performing Attributes**: Gender, Expression, Accessories (90%+)
- **Challenging Attributes**: Age (70-80%), Eye Color (75-85%)

## Notes

- The self-consistency check requires Groq API access
- Test set generation requires CUDA GPU
- Evaluation can take time depending on test set size
- Results are deterministic when using fixed seeds

