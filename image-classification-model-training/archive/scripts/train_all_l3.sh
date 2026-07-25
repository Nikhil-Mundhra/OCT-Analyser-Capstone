#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "================================================================="
echo "  Starting automated L3 Specialist Training Orchestration"
echo "================================================================="

# Ensure we are operating in the image-classification-model-training directory
# This allows the script to be run from anywhere
cd "$(dirname "$0")/.."

# The 5 specialists to train
SPECIALISTS=("Macular" "Diabetic" "Vascular" "Fluid" "Structural")

# Ensure logs directory exists for the orchestration logs
mkdir -p logs/orchestration

# Check for resume flag
RESUME_FLAG=""
if [[ "$1" == "--resume" ]]; then
    echo "🔄 Resume mode enabled."
    RESUME_FLAG="--resume"
fi

for SPEC in "${SPECIALISTS[@]}"; do
    echo ""
    echo "-----------------------------------------------------------------"
    echo "  Starting Training for L3 Specialist: $SPEC"
    echo "-----------------------------------------------------------------"
    
    # If in resume mode, check if this specialist is already completely trained
    # We check for cv_summary.json which is generated at the very end of 5-fold CV
    LOWER_SPEC=$(echo "$SPEC" | tr '[:upper:]' '[:lower:]')
    if [[ "$RESUME_FLAG" == "--resume" ]] && [[ -f "checkpoints/level3_${LOWER_SPEC}/cv_summary.json" ]]; then
        echo "⏭️  $SPEC is already fully trained (cv_summary.json exists). Skipping."
        continue
    fi

    # Create a unique log file for this run
    LOG_FILE="logs/orchestration/train_l3_${SPEC}_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging output to: $LOG_FILE"
    
    # Run the training script wrapped in caffeinate to prevent macOS App Nap / sleep
    # - KMP_DUPLICATE_LIB_OK=TRUE is required on macOS to prevent OpenMP crashes
    # - `tee` streams output to the console and writes it to the log file
    KMP_DUPLICATE_LIB_OK=TRUE caffeinate -i ../.venv/bin/python scripts/train_level3.py --specialist "$SPEC" $RESUME_FLAG | tee "$LOG_FILE"
    
    # PIPESTATUS captures the exit code of the python script before tee
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "❌ Error encountered while training $SPEC. Halting orchestration."
        exit 1
    fi
    echo "✅ Successfully completed training for $SPEC."
done

echo ""
echo "================================================================="
echo "  🎉 All L3 Specialist models have been successfully trained!"
echo "================================================================="

# Send an email notification if the script exists
if [ -f "scripts/send_email.py" ]; then
    echo "Attempting to send completion email notification..."
    # Suppress error if email configuration is not fully set up yet
    ../.venv/bin/python scripts/send_email.py --subject "OCT Pipeline: L3 Specialist Training Complete" --body "All 5 L3 specialist models (Macular, Diabetic, Vascular, Fluid, Structural) have finished training successfully." || echo "Note: Email notification failed (possibly unconfigured)."
fi
