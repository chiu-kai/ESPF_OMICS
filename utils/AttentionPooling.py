import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):  # x: (bsz, seq_len, input_dim)
        attn_scores = self.attn(x).squeeze(-1)  # (bsz, seq_len)
        attn_weights = F.softmax(attn_scores, dim=1)  # (bsz, seq_len)
        pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)  # (bsz, input_dim)
        return pooled, attn_weights


class DrugOmicsFusionModule(nn.Module):
    def __init__(self, include_omics, drugcell_dim=136, omics_dim=128):
        super().__init__()
        self.include_omics = include_omics

        self.omics_attn_poolers = nn.ModuleDict({
            omic: AttentionPooling(input_dim=omics_dim) for omic in include_omics
        })

        self.drugcell_attn_pooler = AttentionPooling(input_dim=drugcell_dim)

        # Example final classifier
        input_dim = len(include_omics) * omics_dim + drugcell_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # For storing attention weights
        self.latest_attention_weights = {}

    def forward(self, omic_embeddings_ls, append_embeddings):
        # omic_embeddings_ls: list of (bsz, c, 128)
        # append_embeddings: (bsz, 50+c, 136) after attention

        omics_pooled = []
        omics_attn_weights = {}

        for i, omic in enumerate(self.include_omics):
            omic_embed = omic_embeddings_ls[i]  # (bsz, ci, 128)
            pooled, attn_w = self.omics_attn_poolers[omic](omic_embed)
            omics_pooled.append(pooled)  # (bsz, 128)
            omics_attn_weights[omic] = attn_w  # (bsz, ci)

        drugcell_pooled, drugcell_attn_weights = self.drugcell_attn_pooler(append_embeddings)

        # concat all pooled vectors
        concat = torch.cat(omics_pooled + [drugcell_pooled], dim=1)  # (bsz, total_dim)

        # Save attention weights
        self.latest_attention_weights = {
            'drugcell': drugcell_attn_weights,
            **omics_attn_weights
        }

        return self.classifier(concat)

    def visualize_attention(self, sample_idx=0):
        attn_dict = self.latest_attention_weights
        for omic_name, attn in attn_dict.items():
            weights = attn[sample_idx].detach().cpu().numpy()  # shape: (seq_len,)
            plt.figure(figsize=(10, 1))
            sns.heatmap(weights[np.newaxis, :], cmap="viridis", cbar=True, xticklabels=False)
            plt.title(f"{omic_name} Attention (sample {sample_idx})")
            plt.yticks([])
            plt.show()