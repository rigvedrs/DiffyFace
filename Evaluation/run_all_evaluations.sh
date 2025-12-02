#!/bin/bash
# Run all evaluation scripts in sequence
# This script automates the complete evaluation pipeline

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$PROJECT_ROOT/Evaluation"
OUTPUT_DIR="$EVAL_DIR/test_set_output"

echo "=========================================="
echo "DiffyFace Evaluation Pipeline"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Generate test set
echo "Step 1: Generating test set..."
echo "----------------------------------------"
python "$EVAL_DIR/generate_test_set.py" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples 300 \
    --seed 42 \
    --steps 50

if [ $? -ne 0 ]; then
    echo "Error: Test set generation failed"
    exit 1
fi

echo ""
echo "Step 1 complete!"
echo ""

# Step 2: Self-consistency evaluation (requires Groq API)
if [ -z "$GROQ_API_KEY" ]; then
    echo "Warning: GROQ_API_KEY not set. Skipping self-consistency evaluation."
    echo "Set GROQ_API_KEY environment variable to run this step."
else
    echo "Step 2: Running self-consistency evaluation..."
    echo "----------------------------------------"
    python "$EVAL_DIR/self_consistency_check.py" \
        --test-set "$OUTPUT_DIR/test_set_metadata.json" \
        --output "$OUTPUT_DIR/self_consistency_results.json"
    
    if [ $? -ne 0 ]; then
        echo "Warning: Self-consistency evaluation failed (may need Groq API key)"
    else
        echo ""
        echo "Step 2 complete!"
    fi
fi

echo ""

# Step 3: Attribute evaluation
echo "Step 3: Running attribute evaluation..."
echo "----------------------------------------"
python "$EVAL_DIR/attribute_evaluation.py" \
    --test-set "$OUTPUT_DIR/test_set_metadata.json" \
    --output "$OUTPUT_DIR/attribute_evaluation_results.json"

if [ $? -ne 0 ]; then
    echo "Error: Attribute evaluation failed"
    exit 1
fi

echo ""
echo "Step 3 complete!"
echo ""

# Step 4: Generate report
echo "Step 4: Generating evaluation report..."
echo "----------------------------------------"
python "$EVAL_DIR/generate_evaluation_report.py" \
    --results-dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/evaluation_report.md"

if [ $? -ne 0 ]; then
    echo "Warning: Report generation failed"
else
    echo ""
    echo "Step 4 complete!"
fi

echo ""
echo "=========================================="
echo "Evaluation Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  - test_set_metadata.json"
echo "  - self_consistency_results.json (if Groq API available)"
echo "  - attribute_evaluation_results.json"
echo "  - evaluation_report.md"
echo "  - Generated images (test_*.png)"
echo ""

