import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from torch.optim.lr_scheduler import CosineAnnealingLR
import seaborn as sns
import matplotlib.colors as mcolors
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.colors as mcolors
import seaborn as sns
import pickle
import joblib
import argparse

from OrganizedClassifyNetwork import CNNClassifier, BN_CNNClassifier, DeepCNNClassifier, BN_DeepCNNClassifier, MoreDeeperCNNClassifier, BN_MoreDeeperCNNClassifier,  SuperDeeperCNNClassifier, BN_SuperDeeperCNNClassifier, BasicBlock, Bottleneck, ResNet1D 
from OrganizedClassifyutilities import resnet18_1d, resnet34_1d, resnet50_1d, resnet101_1d, resnet152_1d, WarmupCosineLR, CustomDataset

def get_model(model_name, num_classes, dropout_prob, sequence_length, device):
    if model_name == 'CNNClassifier':
        return CNNClassifier(num_classes, dropout_prob, sequence_length).to(device)
    elif model_name == 'BN_CNNClassifier':
        return BN_CNNClassifier(num_classes, dropout_prob, sequence_length).to(device)
    elif model_name == 'DeepCNNClassifier':
        return DeepCNNClassifier(num_classes, dropout_prob, sequence_length).to(device)
    elif model_name == 'BN_DeepCNNClassifier':
        return BN_DeepCNNClassifier(num_classes, dropout_prob, sequence_length).to(device)
    elif model_name == 'MoreDeeperCNNClassifier':
        return MoreDeeperCNNClassifier(num_classes, dropout_prob, sequence_length).to(device)
    elif model_name == 'BN_MoreDeeperCNNClassifier':
        return BN_MoreDeeperCNNClassifier(num_classes, dropout_prob, sequence_length).to(device)
    elif model_name == 'SuperDeeperCNNClassifier':
        return SuperDeeperCNNClassifier(num_classes, dropout_prob, sequence_length).to(device)
    elif model_name == 'BN_SuperDeeperCNNClassifier':
        return BN_SuperDeeperCNNClassifier(num_classes, dropout_prob, sequence_length).to(device)
    elif model_name == 'resnet18_1d':
        return resnet18_1d(num_classes=num_classes).to(device)
    elif model_name == 'SVC':
        return SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
    elif model_name == 'RandomForestClassifier':
        return RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
def save_loss_plot(train_losses, val_losses, weightfile):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Save the plot using the same base name as the weightfile but with .png extension
    save_path = 'LossCurve'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_filename = os.path.join(save_path, weightfile) + ".png"
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to avoid displaying it
    print(f"Loss plot saved to {plot_filename}")

def pca(train_features, train_labels, weightfile):
    n_components = min(512, min(train_features.shape))
    pca = PCA(n_components=n_components)
    pca.fit(train_features)
    reduced_data = pca.transform(train_features)
    # print(reduced_data.shape)

    loadings = pca.components_.T  # shape  (n_features, n_components)

    wave_numbers = np.linspace(300, 1900, train_features.shape[1])

    save_path = 'pca'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(10, 6))
    for i in range(3): 
        plt.plot(wave_numbers, loadings[:, i], label=f'PC{i+1}')
    plt.xlabel('Wave Number')
    plt.ylabel('Contribution')
    plt.title('Contribution of Each Wave Number to Principal Components')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'{weightfile}_contributions.png'))
    plt.close()

    explained_variance_ratios = pca.explained_variance_ratio_

    # for i, ratio in enumerate(explained_variance_ratios):
    #     print(f"Principal Component {i+1}: {ratio*100:.2f}%")

    categories = np.unique(train_labels)
    label_mapping = {0: 'Unripe', 1: 'Ripe', 2: 'Overripe'}
    colors = ['#B8D7B3', '#59932D', '#414114']  
    category_colors = {category: colors[i] for i, category in enumerate(categories)}

    # 2D PCA
    plt.figure(figsize=(8, 6))
    for category in categories:
        indices = train_labels == category
        plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], c=category_colors[category], label=category, alpha=0.6)
    plt.xlabel(f'Principal Component 1 ({explained_variance_ratios[0]*100:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({explained_variance_ratios[1]*100:.2f}%)')
    plt.title('PCA of Feature Data')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'{weightfile}_2d.png'))
    plt.close()

    # 3D PCA 
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')

    #  3D PCA 
    for category in categories:
        indices = train_labels == category
        ax.scatter(reduced_data[indices, 0], reduced_data[indices, 1], reduced_data[indices, 2], 
                   c=category_colors[category], label=label_mapping[category], alpha=0.6)

    #  Arial， 14
    ax.set_xlabel(f'PC1 ({explained_variance_ratios[0]*100:.1f}%)', weight='bold', fontsize=14, fontname='Arial')
    ax.set_ylabel(f'PC2 ({explained_variance_ratios[1]*100:.1f}%)', weight='bold', fontsize=14, fontname='Arial')
    ax.set_zlabel(f'PC3 ({explained_variance_ratios[2]*100:.1f}%)', weight='bold', fontsize=14, fontname='Arial')

    #  ticks Arial， 14
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.zaxis.set_tick_params(labelsize=14)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.78, 0.75), fontsize=14, prop={'family': 'Arial'})

    #  3D 
    plt.savefig(os.path.join(save_path, f'{weightfile}_3d.png'))
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='dataset name')

parser.add_argument('--oversample_ratio', type=float, default = 1.0, help='Comma-separated list of oversample ratios for each class')
parser.add_argument('--undersample_ratio', type=float, default= 1.0, help='Comma-separated list of undersample ratios for each class')

parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout probability')
parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
parser.add_argument('--sequence_length', type=int, default=1024, help='Length of input sequence')
parser.add_argument('--model', type=str, required=True, 
                    choices=[
                        'CNNClassifier', 'BN_CNNClassifier', 'DeepCNNClassifier', 'BN_DeepCNNClassifier',
                        'MoreDeeperCNNClassifier', 'BN_MoreDeeperCNNClassifier', 'SuperDeeperCNNClassifier',
                        'BN_SuperDeeperCNNClassifier', 'resnet18_1d', 'SVC', 'RandomForestClassifier'
                    ],
                    help='Choose the model to use for training')
parser.add_argument('--epoch', type=int, default=10, help='epoch number')
parser.add_argument('--weightfile', type=str, required=True, help='Filename to save the trained model weights.')

args = parser.parse_args()

dataset = args.dataset # dataset name
oversample_ratios = args.oversample_ratio # oversample ratio for each class
undersample_ratios = args.undersample_ratio # undersample ratios for each class
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = get_model(args.model, args.num_classes, args.dropout_prob, args.sequence_length, device) # model type
num_epochs = args.epoch  # Number of training epochs
weightfile = args.weightfile # weight file save path


with open(dataset, 'rb') as f:
    data = pickle.load(f)

train_features = data['train_features']
train_labels = data['train_labels']

val_features = data['val_features']
val_labels = data['val_labels']

# print(train_features.shape)

# get PCA figure
pca(train_features, train_labels, weightfile)


# synthetic data
# record the number
train_labels_numpy = train_labels.numpy()
class_counts = Counter(train_labels_numpy)

majority_class = max(class_counts, key=class_counts.get)
minority_classes = [cls for cls in class_counts if cls != majority_class]

sampling_strategy = {cls: max(class_counts[cls], int(class_counts[cls] * oversample_ratios)) for cls in minority_classes}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_smote, y_smote = smote.fit_resample(train_features, train_labels)

sampling_strategy = {majority_class: int(class_counts[majority_class] * undersample_ratios)}
undersample = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
X_undersample, y_undersample = undersample.fit_resample(X_smote, y_smote)


trainf = torch.tensor(X_undersample, dtype=torch.float32)
trainl = torch.tensor(y_undersample, dtype=torch.long)
print("Original label distribution:", dict(zip(*np.unique(train_labels, return_counts=True))))
print("Resampled label distribution:", dict(zip(*np.unique(y_undersample, return_counts=True))))
print("-" * 50)

# augmentation data

# import random
# NUM_OF_SHIFT_PER_IMAGE = 6
# NUM_OF_NOISE_PER_IMAGE = 1
# SHIFT = 0.2
# SIGMA = 0.00001

 # Shift operation
# def shift(X, Y): 
#     X_pre = []
#     Y_pre = []
#     length_x = SHIFT
#     for m in range(len(X)):
#         X_pre.append(X[m])
#         Y_pre.append(Y[m])
#     for i in range(len(X)):
#         for n in range(NUM_OF_SHIFT_PER_IMAGE):
#             x_aug = []
#             for j in range(len(X[i])): 
#                 x = X[i][j] + length_x * (n + 1)  # 只向一个方向移动 # Shift in one direction only
#                 x_aug.append(x)
#             X_pre.append(x_aug)
#             Y_pre.append(Y[i])
#     return np.array(X_pre), np.array(Y_pre)

 # Noise operation
# def noise(X, Y):
#     X_pre = []
#     Y_pre = []
#     mu = 0
#     sigma = SIGMA
#     for m in range(len(X)):
#         X_pre.append(X[m])
#         Y_pre.append(Y[m])
#     for i in range(len(X)):
#         for n in range(NUM_OF_NOISE_PER_IMAGE):
#             x_aug = []
#             for j in range(len(X[i])):           
#                 x = X[i][j] + random.gauss(mu, sigma)
#                 x_aug.append(x)
#             X_pre.append(x_aug)
#             Y_pre.append(Y[i])
#     return np.array(X_pre), np.array(Y_pre)

 # Perform shift operation
# X_shifted, Y_shifted = shift(train_features_norm.tolist(), trainl.tolist())

 # Perform noise operation
# train_features_norm_aug, trainl_aug = noise(X_shifted, Y_shifted)

 # normalize

scaler = Normalizer(norm='max')

train_features_norm = scaler.fit_transform(trainf)
val_features = scaler.transform(val_features)

scaler_path = 'norm_scaler'
if not os.path.exists(scaler_path):
    os.mkdir(scaler_path)
with open(f'{scaler_path}/{weightfile}.pkl', 'wb') as f:
    pickle.dump(scaler, f)




if not args.model =='SVC' or args.model == 'RandomForestClassifier':
     # Create datasets and data loaders
    train_dataset = CustomDataset(train_features_norm, trainl)
    # train_dataset = CustomDataset(train_features_norm_aug, trainl_aug)
    val_dataset = CustomDataset(val_features, val_labels)


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Hyperparameter settings
    learning_rate = 0.0001   # Learning rate
    warmup_epochs = 10   # Warmup epochs




     # Calculate class weights
    # total_samples = len(train_labels)
    # class_counts = np.bincount(train_labels)
    # class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
    # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
     # Use weighted cross-entropy loss
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs, num_epochs, min_lr=learning_rate*0.1)

    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    patience = 10   # Set patience for early stopping
    trigger_times = 0

     # Train the model
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device)
            labels = labels.to(device)

             # Zero the parameter gradients
            optimizer.zero_grad()

            # print(inputs.shape)
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

             # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        model.eval()   # Set model to evaluation mode
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.unsqueeze(1)
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_loss)

        print('Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        # scheduler.step(val_loss)
        scheduler.step(epoch)

         # Early stopping mechanism
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
             # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            trigger_times += 1
            print(f'EarlyStopping trigger times: {trigger_times}')
            
            if trigger_times >= patience:
                print('Early stopping!')
                model.load_state_dict(torch.load('best_model.pth'))
                os.remove('best_model.pth')
                break

    save_loss_plot(train_losses, val_losses, weightfile)

    # torch.save(model.state_dict(), 'model_weights.pth')
    weight_path = 'weightfiles'
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    torch.save(model, f'{weight_path}/{weightfile}.pth')

else:
    model.fit(train_features_norm, trainl)
    weight_path = 'weightfiles'
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    joblib.dump(model, f'{weight_path}/{weightfile}.pkl')

