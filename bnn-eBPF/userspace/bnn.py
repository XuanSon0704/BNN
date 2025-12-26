import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json  

warnings.filterwarnings('ignore')

# --- CONFIGURE ---
DATA_CSV = '/home/sonnguyen/workspace/BNN/bnn-eBPF/userspace/DDOS2017.csv'
BATCH_SIZE = 2048
NUM_EPOCHS = 20
LR = 1e-4
HIDDEN = 64
OUT_DIR = '../results'
os.makedirs(OUT_DIR, exist_ok=True)

SELECTED_FEATURES = [
    'Avg Fwd Segment Size', 
    'Init_Win_bytes_forward',
    'Flow Packets/s',
    'Fwd Packets/s',      
    'Average Packet Size',
    'Total Length of Fwd Packets',
    'Subflow Fwd Bytes',
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
]


# --- BNN ARCH ---
class BinarizeActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input

class Binarize(nn.Module):
    def forward(self, input):
        return BinarizeActivation.apply(input)

class BinarizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        W_b = BinarizeActivation.apply(self.weight)
        return F.linear(input, W_b, bias=None)

class BNN_MLP(nn.Module):
    def __init__(self, n_features, n_hidden=64):
        super().__init__()
        self.hidden1 = nn.Sequential(
            BinarizedLinear(n_features, n_hidden),
            nn.BatchNorm1d(n_hidden),
            Binarize()
        )
        self.hidden2 = nn.Sequential(
            BinarizedLinear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            Binarize()
        )
        self.output_layer = nn.Sequential(
            BinarizedLinear(n_hidden, 1),
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output_layer(x)
        return x

def main():
    
    df = pd.read_csv(DATA_CSV)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    df.columns = df.columns.str.strip()
    missing_cols = [c for c in SELECTED_FEATURES if c not in df.columns]

    y = df['Label']
    y_mapped = y.map({'BENIGN': -1, 'DDoS': 1}).astype(np.float32)

    X = df[SELECTED_FEATURES]
    feature_means = X.mean(axis=0).values.astype(np.float32)
    
    with open(os.path.join(OUT_DIR, 'feature_means.json'), 'w') as f:
        json.dump(feature_means.tolist(), f)
        
    X_binarized = np.where(X.values > feature_means, 1.0, -1.0).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_binarized, y_mapped.values,
        test_size=0.2, random_state=42, stratify=y_mapped.values
    )
    f_vals, _ = f_classif(X_train, y_train)
    
    feat_importances = pd.DataFrame({'Feature': SELECTED_FEATURES, 'F_Score': f_vals})
    feat_importances = feat_importances.sort_values(by='F_Score', ascending=False)
    
    sorted_names = feat_importances['Feature'].tolist()
    sorted_scores = feat_importances['F_Score'].tolist()
    
    plt.figure(figsize=(10, 6))
    # Vẽ bar chart (Horizontal bar plot nhìn sẽ rõ tên hơn)
    sns.barplot(x=sorted_scores, y=sorted_names, palette='viridis')
    plt.xlabel('F-Score')
    plt.title('Top Selected Features ')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ingress_fscore.png'), dpi=200)
    plt.close()
    
    
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test) 
    y_test_tensor = torch.from_numpy(y_test)

    n_features = X_train.shape[1] # Cập nhật số features thực tế
    
    # CREATE DATALOADERS 
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- INIT MODEL ---
    model = BNN_MLP(n_features, n_hidden=HIDDEN)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # --- TRAIN SETUP ---
    def square_hinge_loss(y_pred, y_true):
        return torch.mean(torch.clamp(1 - y_true * y_pred, min=0) ** 2)


    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_prec': [], 'val_rec': [], 'val_f1': []
    }
    
        # ===== TRAIN LOOP =====
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for X_batch, y_batch in tqdm(
            train_loader, desc=f"Train E{epoch+1}/{NUM_EPOCHS}"
        ):
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = square_hinge_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            # clamp weights for BNN
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, BinarizedLinear):
                        if hasattr(module, 'weight'):
                            module.weight.clamp_(-1, 1)

            total_loss += loss.item() * X_batch.size(0)
            preds = y_pred.sign()
            total_correct += (preds == y_batch).sum().item()
            total_samples += X_batch.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_acc)

        # ===== VALIDATION =====
        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch).squeeze()
                loss = square_hinge_loss(y_pred, y_batch)
                total_val_loss += loss.item() * X_batch.size(0)

                preds = y_pred.sign().cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(y_batch.cpu().numpy().tolist())

        avg_val_loss = total_val_loss / len(all_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        scheduler.step(avg_val_loss)
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_prec'].append(precision)
        history['val_rec'].append(recall)
        history['val_f1'].append(f1)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} "
            f"train_loss={avg_loss:.4f} train_acc={avg_acc:.4f} | "
            f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f} "
            f"prec={precision:.4f} rec={recall:.4f} f1={f1:.4f}"
        )


    plt.figure(figsize=(12, 6))
    plt.bar(sorted_names, sorted_scores, color='skyblue')
    plt.xlabel('Features Selected', fontsize=12)
    plt.ylabel('F-score', fontsize=12)
    plt.title(f'Top Selected Features ', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fscore_plot_path = os.path.join(OUT_DIR, 'ingress_fscore.png')
    plt.savefig(fscore_plot_path, dpi=200)
    plt.close() 


    # ===== FINAL EVALUATION =====
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor).squeeze().cpu().numpy()
        preds = np.sign(logits)
        y_true_str = np.where(y_test_tensor.numpy() == 1, 'DDoS', 'BENIGN')
        y_pred_str = np.where(preds == 1, 'DDoS', 'BENIGN')

    report = classification_report(
        y_true_str, y_pred_str,
        target_names=['BENIGN', 'DDoS'],
        zero_division=0
    )
    cm = confusion_matrix(y_true_str, y_pred_str, labels=['BENIGN', 'DDoS'])

    print("\n--- Final Classification Report ---\n")
    print(report)

    # ===== SAVE RESULTS =====
    epoch_ticks = range(0, NUM_EPOCHS + 1, 2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Đồ thị 1: Loss 
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='orange')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_xticks(epoch_ticks)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Đồ thị 2: Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', color='green')
    axes[1].plot(history['val_acc'], label='Val Acc', color='red')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xticks(epoch_ticks)
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Đồ thị 3: Validation Metrics 
    axes[2].plot(history['val_prec'], label='Precision', color='purple')
    axes[2].plot(history['val_rec'], label='Recall', color='brown')
    axes[2].plot(history['val_f1'], label='F1-Score', color='teal', linewidth=2)
    axes[2].set_title('Validation Metrics')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score')
    axes[2].set_xticks(epoch_ticks)
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'training_curves.png'), dpi=200)
    plt.close() 


    with open(os.path.join(OUT_DIR, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Lưu model
    MODEL_PATH = os.path.join(OUT_DIR, 's')
    torch.save(model.state_dict(), MODEL_PATH)

    # Lưu report
    with open(os.path.join(OUT_DIR, 'final_report.txt'), 'w') as f:
        f.write(report)
        f.write('\nConfusion Matrix:\n')
        f.write(str(cm))


if __name__ == "__main__":
    main()