# train_flat.py
from multimodal_base import (
    nn,
    torch,
    PretrainedBackbones,
    CONFIG,
    setup_data_and_train,
)


# ==========================================
# FLAT FUSION MODEL
# ==========================================


class FlatFusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbones = PretrainedBackbones()

        self.common_dim = CONFIG["common_dim"]
        self.txt_proj = nn.Linear(self.backbones.text_dim, self.common_dim)
        self.aud_proj = nn.Linear(self.backbones.audio_dim, self.common_dim)
        self.vid_proj = nn.Linear(self.backbones.video_dim, self.common_dim)

        self.dropout = nn.Dropout(0.3)
        # Fusion is concatenation (3 * common_dim)
        self.classifier = nn.Linear(self.common_dim * 3, num_classes)

    def forward(self, input_ids, mask, audio, video):
        # Extract and Project Features
        t = self.dropout(
            self.txt_proj(self.backbones.get_text_features(input_ids, mask))
        )
        a = self.dropout(self.aud_proj(self.backbones.get_audio_features(audio)))
        v = self.dropout(self.vid_proj(self.backbones.get_video_features(video)))

        # Concat & Classify
        fused = torch.cat((t, a, v), dim=1)
        return self.classifier(fused)


if __name__ == "__main__":
    setup_data_and_train(FlatFusionModel, "Flat Fusion Model", "./data/flat_model.pth")
