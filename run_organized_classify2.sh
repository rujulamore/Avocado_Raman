#!/bin/bash


# Process data for analysis
python OrganizedClassifyReaddata.py \
  --batchname B1+B2allnormal \
  --class1 400 \
  --wavenumberRange 135-2500



# Train the model
python OrganizedClassifyMain.py \
  --dataset "data_split_train_val_test/B1+B2allnormal_class1_350_range_135-2500.pkl" \
  --oversample_ratio 1.0 \
  --undersample_ratio 1.0 \
  --num_class 3 \
  --sequence_length 1024 \
  --model resnet18_1d \
  --epoch 10 \
  --weightfile "resnet18_1d_B1+B2_epoch10"

# Test the trained model with machine learning (ML) model
python OrganizedClassifyTest.py \
  --weightfile "weightfiles/resnet18_1d_B1+B2_epoch10.pkl" \
  --scalerfile "norm_scaler/resnet18_1d_B1+B2_epoch10.pkl" \
  --test_data "data_split_train_val_test/B1+B2allnormal_class1_350_range_135-2500.pkl" \
  --outputfile "resnet18_1d_B1+B2_epoch10" \
  --model dl

# Visualize the processed data
python OrganizedClassifyDrawFigures.py \
  --parquet_files "data_split_train_val_test/B1+B2allnormal_class1_350_range_135-2500.parquet" \
  --output_path "visualizations"