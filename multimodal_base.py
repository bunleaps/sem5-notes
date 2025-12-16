import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa
import cv2
import os
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- NEW IMPORTS FOR PRE-TRAINED MODELS ---
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Model
import torchvision.models.video as video_models

# ==========================================
# 1. CONFIG & UTILS (UPDATED)
# ==========================================
CONFIG = {
    "text_model": "distilroberta-base",
    "audio_model": "facebook/wav2vec2-base-960h",
    "video_model": "r3d_18",
    "max_text_len": 50,
    "audio_duration": 3.0,
    "video_frames": 16,
    "batch_size": 4,
    "epochs": 10,  # Increased
    "lr": 1e-5,  # Lowered for fine-tuning
    "common_dim": 256,  # Increased fusion capacity
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Running on {device} ---")

# ==========================================
# 2. DATASET (ADAPTED FOR FOUNDATION MODELS)
# ==========================================


class MultimodalDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.num_classes = len(self.data["label_idx"].unique())
        self.audio_sr = 16000  # Wav2Vec2 requires 16kHz

    def __len__(self):
        return len(self.data)

    def load_audio_waveform(self, path):
        target_len = int(CONFIG["audio_duration"] * self.audio_sr)

        if path == "N/A" or not isinstance(path, str) or not os.path.exists(path):
            return torch.zeros((target_len,))

        try:
            y, sr = librosa.load(
                path, sr=self.audio_sr, duration=CONFIG["audio_duration"]
            )
            y_tensor = torch.tensor(y, dtype=torch.float32)

            if y_tensor.shape[0] < target_len:
                padding = torch.zeros((target_len - y_tensor.shape[0],))
                y_tensor = torch.cat((y_tensor, padding))
            else:
                y_tensor = y_tensor[:target_len]
            return y_tensor
        except Exception:
            return torch.zeros((target_len,))

    def load_video_clip(self, path):
        C, F, H, W = 3, CONFIG["video_frames"], 112, 112

        if path == "N/A" or not isinstance(path, str) or not os.path.exists(path):
            return torch.zeros((C, F, H, W))

        frames = []
        try:
            cap = cv2.VideoCapture(path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Calculate step to ensure uniform sampling
            step = max(1, total_frames // F)

            count = 0
            while len(frames) < F:
                ret, frame = cap.read()
                if not ret:
                    break
                if count % step == 0:
                    frame = cv2.resize(frame, (H, W))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame / 255.0
                    frames.append(frame)
                count += 1
            cap.release()
        except:
            pass

        if len(frames) < F:
            missing = F - len(frames)
            for _ in range(missing):
                frames.append(np.zeros((H, W, 3)))

        video_tensor = torch.tensor(np.array(frames), dtype=torch.float32).permute(
            3, 0, 1, 2
        )
        return video_tensor

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text_enc = self.tokenizer(
            row["transcript_text"],
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_text_len"],
            return_tensors="pt",
        )
        input_ids = text_enc["input_ids"].squeeze(0)
        attention_mask = text_enc["attention_mask"].squeeze(0)
        audio_wave = self.load_audio_waveform(row["audio_path"])
        video_vol = self.load_video_clip(row["video_path"])
        label = torch.tensor(row["label_idx"], dtype=torch.long)

        return input_ids, attention_mask, audio_wave, video_vol, label


# ==========================================
# 3. PRE-TRAINED FEATURE EXTRACTORS (IMPROVED: Partial Fine-Tuning)
# ==========================================


class PretrainedBackbones(nn.Module):
    def __init__(self):
        super().__init__()

        # --- TEXT: RoBERTa ---
        self.text_model = AutoModel.from_pretrained(CONFIG["text_model"])
        self.text_dim = 768

        # --- AUDIO: Wav2Vec 2.0 ---
        self.audio_model = Wav2Vec2Model.from_pretrained(CONFIG["audio_model"])
        self.audio_dim = 768

        # --- VIDEO: ResNet3D-18 ---
        self.video_model = video_models.r3d_18(
            weights=video_models.R3D_18_Weights.KINETICS400_V1
        )
        self.video_model.fc = nn.Identity()
        self.video_dim = 512

        # 1. FREEZE ALL BY DEFAULT
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.audio_model.parameters():
            param.requires_grad = False
        for param in self.video_model.parameters():
            param.requires_grad = False

        # 2. SELECTIVE UNFREEZING (FINE-TUNING)
        # Text (RoBERTa): Unfreeze the last 4 encoder layers (8-11) and embeddings
        for name, param in self.text_model.named_parameters():
            if "embeddings" in name or any(
                f"encoder.layer.{i}." in name for i in range(8, 12)
            ):
                param.requires_grad = True

        # Audio (Wav2Vec): Unfreeze the feature extractor and last 4 encoder layers (8-11)
        for name, param in self.audio_model.named_parameters():
            if "feature_extractor" in name or any(
                f"encoder.layers.{i}." in name for i in range(8, 12)
            ):
                param.requires_grad = True

        # Video (R3D): Unfreeze the deepest layer ('layer4') and the first conv layer
        for name, param in self.video_model.named_parameters():
            if "layer4" in name or "conv1" in name:
                param.requires_grad = True

    def get_text_features(self, input_ids, mask):
        out = self.text_model(input_ids=input_ids, attention_mask=mask)[0]
        return out[:, 0, :]  # [CLS] token

    def get_audio_features(self, waveforms):
        out = self.audio_model(waveforms).last_hidden_state
        return torch.mean(out, dim=1)  # Mean Pooling

    def get_video_features(self, video_vol):
        return self.video_model(video_vol)


# ==========================================
# 4. TRAINING LOOP (IMPROVED: Best Model Saving & Epoch Saving)
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, epoch_idx):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc=f"Epoch {epoch_idx}", leave=True)

    for ids, mask, aud, vid, lbl in loop:
        ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
        aud, vid = aud.to(device), vid.to(device)

        optimizer.zero_grad()
        outputs = model(ids, mask, aud, vid)
        loss = criterion(outputs, lbl)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for ids, mask, aud, vid, lbl in tqdm(loader, desc="Eval", leave=False):
            ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
            aud, vid = aud.to(device), vid.to(device)

            out = model(ids, mask, aud, vid)
            p = torch.argmax(out, dim=1)
            preds.extend(p.cpu().numpy())
            truths.extend(lbl.cpu().numpy())

    return accuracy_score(truths, preds), f1_score(truths, preds, average="weighted")


def setup_data_and_train(model_cls, name, base_path):
    CSV_PATH = "./data/dataset.csv"
    if not os.path.exists(CSV_PATH):
        return print("Dataset not found")

    # 1. Prepare Data
    df = pd.read_csv(CSV_PATH)
    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df["emotion_label"])

    # Calculate Class Weights (Crucial for F1/CrossEntropyLoss)
    counts = df["label_idx"].value_counts().sort_index().values
    weights = torch.tensor([sum(counts) / c for c in counts], dtype=torch.float32).to(
        device
    )
    # Normalize weights
    weights = weights / weights.sum()

    # Load Transformers Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model"])

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_ds = MultimodalDataset(train_df, tokenizer)
    test_ds = MultimodalDataset(test_df, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )

    # 2. Train and Save
    print(f"\n--- Training {name} ---")
    model = model_cls(num_classes=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Only optimize the fusion layers and unfreezed backbones
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["lr"]
    )

    best_f1 = 0.0
    best_path = base_path.replace(".pth", "_best.pth")

    save_dir = os.path.dirname(base_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for ep in range(CONFIG["epochs"]):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, ep + 1)
        acc, f1 = evaluate(model, test_loader)
        print(f"   Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

        # Save Checkpoint for every epoch
        epoch_path = base_path.replace(".pth", f"_epoch_{ep+1}.pth")
        torch.save(model.state_dict(), epoch_path)
        print(f"   [Checkpoint saved: {epoch_path}]")

        # Track and save the BEST model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            print(f"   [NEW BEST F1! SAVED to {best_path}]")

    print(f"Final Best F1 for {name}: {best_f1:.4f}")
