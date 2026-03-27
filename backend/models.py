import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


# ============================================================
# 1. Wav2Vec2 Detector (Self-supervised Transformer Baseline)
# ============================================================
class AttentivePooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(w * x, dim=1)

class Wav2Vec2SpoofDetector(nn.Module):
    def __init__(self, num_classes=2, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)

        #freeze model
        for param in self.wav2vec.parameters():
            param.requires_grad = False

        hidden = self.wav2vec.config.hidden_size
        self.pool = AttentivePooling(hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        out = self.wav2vec(x).last_hidden_state
        pooled = self.pool(out)
        return self.classifier(pooled)

# ============================================================
# 2. AASIST (SOTA Graph-based Baseline)
# ============================================================

class GraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(out_dim * 2, 1)

    def forward(self, x):
        h = self.fc(x)
        B, N, D = h.shape
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        attn_input = torch.cat([h_i, h_j], dim=-1)
        e = self.attn(attn_input).squeeze(-1)
        alpha = F.softmax(e, dim=-1)
        out = torch.matmul(alpha, h)
        return out

class GraphBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gat = GraphAttention(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        res = x
        x = self.gat(x)
        x = self.norm(x + res)
        return x

class AASISTDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Graph layers
        self.graph1 = GraphBlock(128)
        self.graph2 = GraphBlock(128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.shape
        nodes = x.view(B, C, -1).permute(0, 2, 1)
        nodes = self.graph1(nodes)
        nodes = self.graph2(nodes)
        pooled = nodes.mean(dim=1)
        return self.fc(pooled)

# ============================================================
# 3. CQCC Baseline Detector (Acoustic Feature Baseline)
# ============================================================

class CQCCBaselineDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Input shape expected: (B, 1, 20, T)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

# ============================================================
# 4. Custom Fusional Wav2Vec2 + CQCC with Cross-Attention + Graph
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=2000):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1)]

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        # x1 attends to x2
        q1 = self.norm_q(x1)
        k2 = self.norm_kv(x2)
        v2 = k2
        out1, _ = self.attn1(q1, k2, v2)

        # x2 attends to x1
        q2 = self.norm_q(x2)
        k1 = self.norm_kv(x1)
        v1 = k1
        out2, _ = self.attn2(q2, k1, v1)
        return out1, out2

def align_sequences(x, target_len):
    """Linear interpolation to match sequence lengths"""
    x = x.transpose(1, 2)
    x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
    return x.transpose(1, 2)

class ImprovedWav2Vec2CQCCDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Wav2Vec2
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Freeze the Wav2Vec2 layer so it acts purely as a feature extractor
        for param in self.wav2vec.parameters():
            param.requires_grad = False
            
        dim = self.wav2vec.config.hidden_size

        # CQCC encoder
        self.cqcc_conv = nn.Sequential(
            nn.Conv1d(20, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )

        # Positional Encoding
        self.pos_enc = PositionalEncoding(dim)

        # Bidirectional Cross Attention
        self.cross_attn = BidirectionalCrossAttention(dim)

        # True Graph Transformer Backend (using GAT blocks from AASIST)
        self.graph_layers = nn.ModuleList([
            GraphBlock(dim) for _ in range(3)
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, wav, cqcc):
        if wav.dim() == 3:
            wav = wav.squeeze(1)

        # Wav2Vec2 features
        w2v = self.wav2vec(wav).last_hidden_state  # (B, T_w, D)

        # CQCC features
        if cqcc.dim() == 4:
            cqcc = cqcc.squeeze(1)
        cqcc_feat = self.cqcc_conv(cqcc).transpose(1, 2)  # (B, T_c, D)

        # Align lengths
        cqcc_feat = align_sequences(cqcc_feat, w2v.size(1))

        # Add positional encoding
        w2v = self.pos_enc(w2v)
        cqcc_feat = self.pos_enc(cqcc_feat)

        # Cross attention (bidirectional)
        f1, f2 = self.cross_attn(cqcc_feat, w2v)
        fused = f1 + f2

        # Graph Transformer processing on node sequences
        x = fused
        for layer in self.graph_layers:
            x = layer(x)

        # Global average pooling on the nodes
        pooled = x.mean(dim=1)

        return self.classifier(pooled)
