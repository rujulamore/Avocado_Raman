from OrganizedClassifyutilities import resnet18_1d, resnet34_1d, resnet50_1d, resnet101_1d, resnet152_1d, WarmupCosineLR, CustomDataset
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score

import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
import joblib
import argparse
import os
import warnings
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--weightfile', type=str, required=True, help='Path to the model weight file')
parser.add_argument('--scalerfile', type=str, required=True, help='Path to the scaler file for normalization')
parser.add_argument('--test_data', type=str, required=True, help='Path to the test dataset')
parser.add_argument('--outputfile', type=str, required=True, help='Path to save the output results')
parser.add_argument('--model', type=str, required=True, choices=['dl', 'ml'], help='Indicates whether the model is deep learning (dl) or machine learning (ml)')
args = parser.parse_args()

# Suppress undefined metric warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Create a test folder if it doesn't exist
os.makedirs('testresults', exist_ok=True)
args.outputfile = os.path.join('testresults', args.outputfile)

# Load test data
with open(args.test_data, 'rb') as f:
    data = pickle.load(f)
test_features = data['test_features']
test_labels = data['test_labels']

if args.model == 'dl':
    with open(args.scalerfile, 'rb') as f:
        scaler = pickle.load(f)
    test_features = scaler.transform(test_features)

    test_dataset = CustomDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.weightfile)

    ## test
    model.eval()  # 设置模型为评估模式 # Set model to evaluation mode
    test_running_loss = 0.0
    test_running_corrects = 0
    all_preds = []
    all_labels = []
    all_probs = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  # No gradient calculation for testing
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            test_running_loss += loss.item() * inputs.size(0)
            test_running_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    test_loss = test_running_loss / len(test_loader.dataset)
    test_acc = test_running_corrects.double() / len(test_loader.dataset)
    test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    test_f1_score = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Prepare ROC data
    unique_labels = set(all_labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in unique_labels:
        fpr[str(i)], tpr[str(i)], _ = roc_curve([1 if label == i else 0 for label in all_labels], [prob[i] for prob in all_probs])
        roc_auc[str(i)] = roc_auc_score([1 if label == i else 0 for label in all_labels], [prob[i] for prob in all_probs])

    # Convert results to DataFrame
    output_results = {
        'Test Loss': [test_loss],
        'Accuracy': [test_acc.item()],
        'Precision': [test_precision],
        'Recall': [test_recall],
        'F1 Score': [test_f1_score],
        'Confusion Matrix': [confusion_matrix(all_labels, all_preds).tolist()],
        'Predictions': [all_preds],
        'True Labels': [all_labels],
        'ROC Curve': [{
            'fpr': {str(i): fpr[i].tolist() for i in fpr},
            'tpr': {str(i): tpr[i].tolist() for i in tpr},
            'roc_auc': {str(i): float(roc_auc[i]) for i in roc_auc}
        }]
    }
    df = pd.DataFrame(output_results)

    # Save DataFrame as Parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, args.outputfile + '.parquet')

else: 
    with open(args.scalerfile, 'rb') as f:
        scaler = pickle.load(f)
    test_features = scaler.transform(test_features)

    model = joblib.load(args.weightfile)
    all_probs = model.predict_proba(test_features)
    pred = model.predict(test_features)

    acc = accuracy_score(test_labels, pred)
    pre = precision_score(test_labels, pred, average='macro', zero_division=0)
    recall = recall_score(test_labels, pred, average='macro', zero_division=0)
    f1 = f1_score(test_labels, pred, average='macro', zero_division=0)

    # Prepare ROC data
    unique_labels = set(test_labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in unique_labels:
        fpr[str(i)], tpr[str(i)], _ = roc_curve([1 if label == i else 0 for label in test_labels], [prob[i] for prob in all_probs])
        roc_auc[str(i)] = roc_auc_score([1 if label == i else 0 for label in test_labels], [prob[i] for prob in all_probs])

    # Convert results to DataFrame
    output_results = {
        'Accuracy': [acc],
        'Precision': [pre],
        'Recall': [recall],
        'F1 Score': [f1],
        'Confusion Matrix': [confusion_matrix(test_labels, pred).tolist()],
        'Predictions': [pred.tolist()],
        'True Labels': [test_labels.tolist()],
        'ROC Curve': [{
            'fpr': {str(i): fpr[i].tolist() for i in fpr},
            'tpr': {str(i): tpr[i].tolist() for i in tpr},
            'roc_auc': {str(i): float(roc_auc[i]) for i in roc_auc}
        }]
    }
    df = pd.DataFrame(output_results)

    # Save DataFrame as Parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, args.outputfile + '.parquet')