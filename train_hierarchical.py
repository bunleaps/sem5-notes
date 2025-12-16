# train_hierarchical.py
from multimodal_base import (
    nn,
    torch,
    PretrainedBackbones,
    CONFIG,
    setup_data_and_train,
)


# ==========================================
# HIERARCHICAL FUSION MODEL
# ==========================================


class HierarchicalFusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbones = PretrainedBackbones()

        self.common_dim = CONFIG["common_dim"]
        self.txt_proj = nn.Linear(self.backbones.text_dim, self.common_dim)
        self.aud_proj = nn.Linear(self.backbones.audio_dim, self.common_dim)
        self.vid_proj = nn.Linear(self.backbones.video_dim, self.common_dim)

        self.dropout = nn.Dropout(0.3)

        # Hierarchical Fusion Layers
        # T + A -> Common_Dim
        self.fusion_l1 = nn.Linear(self.common_dim * 2, self.common_dim)
        # (L1 Output) + V -> Common_Dim
        self.fusion_l2 = nn.Linear(self.common_dim * 2, self.common_dim)

        self.classifier = nn.Linear(self.common_dim, num_classes)

    def forward(self, input_ids, mask, audio, video):
        # Extract and Project Features
        t = self.dropout(
            self.txt_proj(self.backbones.get_text_features(input_ids, mask))
        )
        a = self.dropout(self.aud_proj(self.backbones.get_audio_features(audio)))
        v = self.dropout(self.vid_proj(self.backbones.get_video_features(video)))

        # Level 1: Text + Audio
        l1 = torch.cat((t, a), dim=1)
        l1_out = torch.relu(self.fusion_l1(l1))

        # Level 2: L1 + Video
        l2 = torch.cat((l1_out, v), dim=1)
        l2_out = torch.relu(self.fusion_l2(l2))

        return self.classifier(l2_out)


if __name__ == "__main__":
    setup_data_and_train(
        HierarchicalFusionModel, "Hierarchical Fusion Model", "./data/hier_model.pth"
    )
