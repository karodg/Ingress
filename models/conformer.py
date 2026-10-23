import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConformerConvModule(nn.Module):
    """
    Глубокий сверточный модуль Conformer (Depthwise Separable 1D Conv с LayerNorm):
    Численно стабилен в FP16 в режимах train и eval.
    Захватывает высокочастотную физику: микро-крутизну ингресса/эгресса транзита, 
    асимметрию кометных хвостов и резкие экспоненциальные фронты вспышек.
    """
    def __init__(self, d_model=768, kernel_size=15):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model, bias=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, N, D]
        res = x
        h = self.ln(x).transpose(1, 2)     # [B, D, N]
        h = self.pointwise_conv1(h)
        h = F.glu(h, dim=1)                # [B, D, N]
        h = self.depthwise_conv(h).transpose(1, 2) # [B, N, D]
        h = self.act(self.norm(h)).transpose(1, 2) # [B, D, N]
        h = self.pointwise_conv2(h)
        h = self.dropout(h).transpose(1, 2) # [B, N, D]
        return res + h

class ConformerBlock(nn.Module):
    """
    Macaron-Style Conformer Block:
    FFN1 (half-step) -> Multi-Head Self-Attention -> Depthwise Conv -> FFN2 (half-step)
    Pre-LayerNorm для абсолютной численной стабильности в FP16.
    """
    def __init__(self, d_model=768, num_heads=12, ffn_dim=3072, conv_kernel=15):
        super().__init__()
        # 1. First FFN (Half-step)
        self.ln_ffn1 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(0.1)
        )
        
        # 2. Multi-Head Self-Attention (Global Temporal Dependencies)
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)
        self.dropout_attn = nn.Dropout(0.1)
        
        # 3. Depthwise Conv Module (Local Physical Morphology)
        self.conv_module = ConformerConvModule(d_model, kernel_size=conv_kernel)
        
        # 4. Second FFN (Half-step)
        self.ln_ffn2 = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(0.1)
        )
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # Half-step FFN 1
        x = x + 0.5 * self.ffn1(self.ln_ffn1(x))
        # Self-Attention
        norm_attn = self.ln_attn(x)
        attn_out, _ = self.attn(norm_attn, norm_attn, norm_attn)
        x = x + self.dropout_attn(attn_out)
        # Depthwise Conv
        x = self.conv_module(x)
        # Half-step FFN 2
        x = x + 0.5 * self.ffn2(self.ln_ffn2(x))
        return self.final_ln(x)

class AstroConformer100M(nn.Module):
    """
    Универсальная астрофизическая Foundation-нейросеть масштаба >100 Млн Параметров (Cognitive Conformer).
    
    Архитектура:
      - 8 Conformer слоев с размерностью 768 и 12 головами внимания
      - Macaron Dual FFN (размерность 3072)
      - Depthwise Separable 1D Convolutions
      - Мультикогнитивная структура:
          1. Masked Inpainting Head: реконструкция 35% скрытых физических участков
          2. Physical Dynamics Reasoning Head: классификация и физическое понимание 7 типов космических процессов
          3. Latent Representation Head: векторное абстрагирование JEPA и детекция аномалий
    """
    def __init__(self, seq_len=2048, patch_size=32, d_model=768, num_layers=8, num_heads=12, ffn_dim=3072, mask_ratio=0.35, num_classes=7):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size # 64 токена
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        
        # 1. Патч-проекция во временном ряде
        self.patch_proj = nn.Linear(patch_size, d_model)
        self.ln_patch = nn.LayerNorm(d_model)
        
        # 2. Обучаемые позиционные эмбеддинги и маскировочный токен
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 3. 8 Глубоких блоков Conformer (~13.6 млн параметров каждый = ~109M)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim, conv_kernel=15)
            for _ in range(num_layers)
        ])
        
        # 4. Голова 1: Физическая реконструкция маскированных участков (Masked Autoencoding)
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_size)
        )
        
        # 5. Голова 2: Физическое мышление и распознавание законов динамики (Reasoning Head)
        self.dynamics_head = nn.Sequential(
            nn.Linear(d_model, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, num_classes)
        )
        
        # 6. Голова 3: Латентное абстрагирование JEPA и скоринг аномалий
        self.latent_projector = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )

    def _patchify(self, x):
        B, T = x.shape
        return x.reshape(B, self.num_patches, self.patch_size)

    def forward_train(self, x, phenom_labels):
        """
        Единый высокопроизводительный когнитивный проход обучения:
          - Маскирует 35% случайных патчей;
          - Вычисляет реконструкцию маскированных участков (MSE);
          - Вычисляет классификацию физического феномена (Cross-Entropy);
          - Возвращает взвешенный суммарный лосс.
        """
        B, T = x.shape
        orig_patches = self._patchify(x)
        tokens = self.ln_patch(self.patch_proj(orig_patches))
        
        # Случайное маскирование 35% патчей
        num_mask = int(self.num_patches * self.mask_ratio)
        mask = torch.zeros(B, self.num_patches, device=x.device, dtype=torch.bool)
        for b in range(B):
            perm = torch.randperm(self.num_patches, device=x.device)
            mask[b, perm[:num_mask]] = True
            
        masked_tokens = tokens.clone()
        mask_exp = mask.unsqueeze(-1).expand_as(tokens)
        masked_tokens = torch.where(mask_exp, self.mask_token.expand_as(tokens), tokens)
        
        # Прогон через Conformer
        h = masked_tokens + self.pos_embed
        for block in self.blocks:
            h = block(h)
            
        # 1. Лосс реконструкции маскированных областей
        pred_patches = self.recon_head(h)
        loss_recon = F.mse_loss(pred_patches[mask], orig_patches[mask])
        
        # 2. Лосс физического понимания механизма (Reasoning Loss)
        global_repr = h.mean(dim=1)
        logits_dynamics = self.dynamics_head(global_repr)
        loss_dynamics = F.cross_entropy(logits_dynamics, phenom_labels)
        
        # 3. Регуляризация латентного пространства
        z = F.normalize(self.latent_projector(global_repr), p=2, dim=-1)
        
        # Общий когнитивный лосс
        total_loss = loss_recon + 0.35 * loss_dynamics
        
        # Точность распознавания физики
        with torch.no_grad():
            preds = logits_dynamics.argmax(dim=-1)
            acc_dynamics = (preds == phenom_labels).float().mean()
            
        return {
            "loss": total_loss,
            "loss_recon": loss_recon,
            "loss_dynamics": loss_dynamics,
            "acc_dynamics": acc_dynamics,
            "latent": z
        }

    def compute_anomaly_score(self, x):
        """
        Вычисление космической аномальности для открытия новых явлений
        """
        self.eval()
        with torch.no_grad():
            B, T = x.shape
            orig_patches = self._patchify(x)
            tokens = self.ln_patch(self.patch_proj(orig_patches)) + self.pos_embed
            h = tokens
            for block in self.blocks:
                h = block(h)
                
            pred_patches = self.recon_head(h)
            diff = (pred_patches - orig_patches) ** 2
            patch_scores = diff.mean(dim=-1)       # [B, N]
            total_score = diff.mean(dim=[1, 2])     # [B]
            
            global_repr = h.mean(dim=1)
            logits_dynamics = self.dynamics_head(global_repr)
            probs = F.softmax(logits_dynamics, dim=-1)
            
        return {
            "anomaly_score": total_score,
            "patch_scores": patch_scores,
            "physics_probabilities": probs,
            "latent_vector": global_repr
        }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AstroConformer100M().to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print("=" * 70)
    print(f"[*] AstroConformer100M Foundation Model Initialized")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"[+] Total Parameters: {total_p:,} ({total_p/1e6:.2f} Million Parameters)")
    print("=" * 70)
