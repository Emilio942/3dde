#!/bin/bash

# Script for running overnight training
# Logs start and end time for duration tracking

LOG_FILE="training_overnight.log"
CONFIG="experiments/configs/overnight.yaml"

echo "========================================" | tee -a $LOG_FILE
echo "STARTING OVERNIGHT TRAINING" | tee -a $LOG_FILE
echo "Date: $(date)" | tee -a $LOG_FILE
echo "Config: $CONFIG" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

START_TIME=$(date +%s)

# Run training
# Using nohup to ensure it keeps running if the terminal closes (optional, but good practice)
# But here we run it directly as requested.
/home/emilio/Documents/ai/3dde/.venv/bin/python -m experiments.train_with_config --config $CONFIG 2>&1 | tee -a $LOG_FILE

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Calculate hours, minutes, seconds
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

echo "========================================" | tee -a $LOG_FILE
echo "TRAINING COMPLETED" | tee -a $LOG_FILE
echo "Date: $(date)" | tee -a $LOG_FILE
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

echo "Checkpoints saved in checkpoints_overnight/"
