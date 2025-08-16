import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import argparse
from OrganizedClassifyutilities import interpo
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--batchname', type=str, help='batch name')
parser.add_argument('--class1', type=str, help='number of the class1(Ripe)')
parser.add_argument('--wavenumberRange', type=str, help = 'the wavenumber range in Raman figure')


args = parser.parse_args()

batchnum = args.batchname
class_1 = args.class1
start_end = args.wavenumberRange

if class_1 is None:
    class_1 = ''


# Read xlsx file
# batchnum = "B1+B2Classify" # including everything
# batchnum = "B1+B2_normal" # including B1 remained and B2 removed abnormal
# batchnum = "B1+B2allnormal" # including both B1 and B2 removed abnormal data

 
csv_dir = f'dataset/data/{batchnum}/'

# mapping_dict = np.load(f'dataset/mapping_file{batchnum}400.npy', allow_pickle=True).item()
# mapping_dict = np.load(f'dataset/mapping_file{batchnum}350.npy', allow_pickle=True).item()
# mapping_dict = np.load(f'dataset/mapping_file{batchnum}300.npy', allow_pickle=True).item()
mapping_dict = np.load(f'dataset/mapping_file{batchnum}{class_1}.npy', allow_pickle=True).item()


feature_data = [] # Feature list
label_list = []   # Firmness list
inner_images = []  # Inner color of avocado

keys = list(mapping_dict.keys())

for filename in os.listdir(csv_dir): # Iterate over CSV folder
    if os.path.splitext(filename)[0] + ".xlsx" in keys:
        file_path = os.path.join(csv_dir, filename)   # Change to CSV file path
        df = pd.read_csv(file_path)  
        features = df.iloc[:, 2].values  # Represents a single Raman feature
        
        feature_data.append(features) 
        feature_key = os.path.splitext(filename)[0]
        label = mapping_dict.get(feature_key + ".xlsx")
        label_list.append(label/4.4482)

df = pd.read_csv(f'dataset/label/{batchnum}{class_1}.csv')

inner_images = df.iloc[:, 2].values
# label_array = df.iloc[:, 5].values 
# label_array = df.iloc[:, 11].values
label_array = df['Label'].values

 # Read CSV file
df = pd.read_csv(f'dataset/data/B1+B2Classify/RS20240205130708_P19100983.csv')
raman = df.iloc[:,1].values
ramans = [raman for _ in range(len(label_list))]

 # Count the number of each element in label_array
counter = Counter(label_array)
 # Output the count of elements equal to 0, 1, and 2
count_0 = counter[0]
count_1 = counter[1]
count_2 = counter[2]
print(f'Number of elements under ripe: {count_0}')
print(f'Number of elements ripe: {count_1}')
print(f'Number of elements over ripe: {count_2}')

 # If not slicing 135-2500
if start_end == "135-2500":
    cut_feature_data = feature_data
    cut_ramans = ramans
    start = 135
    end = 2500

# Slice from 300 - 1900
elif start_end == "300-1900":
    cut_feature_data = [row[26:327] for row in feature_data]
    cut_ramans = [row[26:327] for row in ramans]
    start = 300
    end = 1900 

# Slice from 400 - 1800
elif start_end == "400-1800":
    cut_feature_data = [row[42:303] for row in feature_data]
    cut_ramans = [row[42:303] for row in ramans]
    start = 400
    end = 1800 

# interpolation data

points = 1024

cut_feature_data = torch.tensor(np.array(cut_feature_data), dtype=torch.float32)

 # Interpolation
x_data = interpo(cut_ramans, cut_feature_data, start, end, points)  # Interpolated features

tensorFeatures = torch.tensor(x_data, dtype=torch.float32)
tensorLabels = torch.tensor(label_array, dtype=torch.long)

# split dataset

 # Split dataset in a 6:1:3 ratio
train_features, temp_features, train_labels, temp_labels = train_test_split(tensorFeatures, tensorLabels, test_size=0.4, random_state=42)
val_features, test_features, val_labels, test_labels = train_test_split(temp_features, temp_labels, test_size=0.75, random_state=42)



filename = f"{batchnum}_class1_{class_1}_range_{start_end}.pkl"
save_path = 'data_split_train_val_test'
if not os.path.exists(save_path):
    os.makedirs(save_path)

with open(os.path.join(save_path,filename), 'wb') as f:
    pickle.dump({
        'train_features': train_features,
        'train_labels': train_labels,
        'val_features': val_features,
        'test_features': test_features,
        'val_labels': val_labels,
        'test_labels': test_labels
    }, f)
