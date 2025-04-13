#!/bin/bash

# Set dataset and weight file paths
dataset_path="data_split_train_val_test/B1+B2allnormal_class1_350_range_135-2500.pkl"
weightfile="resnet18_1d_B1+B2_epoch10"
scalerfile="norm_scaler/${weightfile}.pkl"
test_data_path="data_split_train_val_test/B1+B2allnormal_class1_350_range_135-2500.pkl"
outputfile="test"
output_directory="testresults/combined_visualizations"

# Run training
python OrganizedClassifyMain.py \
  --dataset "$dataset_path" \
  --oversample_ratio 1.0 \
  --undersample_ratio 1.0 \
  --dropout_prob 0.5 \
  --num_classes 3 \
  --sequence_length 1024 \
  --model "resnet18_1d" \
  --epoch 10 \
  --weightfile "$weightfile"

# Run testing
python OrganizedClassifyTest.py \
  --weightfile "weightfiles/${weightfile}.pth" \
  --scalerfile "$scalerfile" \
  --test_data "$test_data_path" \
  --outputfile "$outputfile" \
  --model "dl"

# Generate combined ROC curves
python OrganizedClassifyDrawFigures.py \
  --parquet_files "testresults/${outputfile}.parquet" \
  --output_path "$output_directory"