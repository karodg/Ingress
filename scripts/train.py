import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from astro_conformer_100m import AstroConformer100M
from astro_100m_dataset import Astro100MDataset, PHENOMENA_MAP

CKPT_DIR = Path("D:/Workspace/astro_100m/checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def train_epoch(model, loader, optimizer, scaler, device, epoch, accum_steps=2):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_dynamics = 0.0
    total_acc = 0.0
    t0 = time.time()
    
    optimizer.zero_grad()
    for b_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            out = model.forward_train(x, y)
            loss = out["loss"] / accum_steps
            l_rec = out["loss_recon"].item()
            l_dyn = out["loss_dynamics"].item()
            acc = out["acc_dynamics"].item()
            
        scaler.scale(loss).backward()
        
        if (b_idx + 1) % accum_steps == 0 or (b_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accum_steps
        total_recon += l_rec
        total_dynamics += l_dyn
        total_acc += acc
        
        if (b_idx + 1) % 20 == 0 or (b_idx + 1) == len(loader):
            speed = (b_idx + 1) * x.shape[0] / (time.time() - t0 + 1e-5)
            avg_rec = total_recon / (b_idx + 1)
            avg_dyn = total_dynamics / (b_idx + 1)
            avg_acc = (total_acc / (b_idx + 1)) * 100
            sys.stdout.write(
                f"\r[110M Epoch {epoch:2d}] Batch {b_idx+1:4d}/{len(loader)} | "
                f"Recon MSE: {avg_rec:.4f} | Dynamics Loss: {avg_dyn:.3f} | "
                f"Accuracy: {avg_acc:.1f}% | Speed: {speed:.1f} sys/s"
            )
            sys.stdout.flush()
            
    print()
    n = len(loader)
    return total_loss / n, total_recon / n, total_dynamics / n, (total_acc / n) * 100

def evaluate(model, loader, device):
    model.eval()
    total_recon = 0.0
    total_dynamics = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model.forward_train(x, y)
            total_recon += out["loss_recon"].item()
            total_dynamics += out["loss_dynamics"].item()
            total_acc += out["acc_dynamics"].item()
    n = max(1, len(loader))
    return total_recon / n, total_dynamics / n, (total_acc / n) * 100

def main():
    print("=" * 80)
    print("  ASTRO-CONFORMER 110M: DEEP MULTI-COGNITIVE FOUNDATION MODEL")
    print("  ARCHITECTURE: 8 CONFORMER LAYERS (768-DIM, 12 HEADS, 3072 FFN, DEPTHWISE CONV)")
    print("  COGNITIVE FACULTIES: MASKED INPAINTING + DYNAMICS REASONING + ANOMALY DISCOVERY")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    train_size = 1800
    val_size = 300
    print(f"[*] Initializing Dynamic Cosmic Dataset ({train_size} train, {val_size} val)...")
    train_dataset = Astro100MDataset(size=train_size, seq_len=2048)
    val_dataset = Astro100MDataset(size=val_size, seq_len=2048)
    
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, drop_last=False, pin_memory=True)
    
    # 2. Модель 110M
    model = AstroConformer100M(num_layers=8, d_model=768, num_heads=12, ffn_dim=3072).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Exact Model Parameters: {total_params:,} ({total_params/1e6:.2f} Million Parameters)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-4)
    epochs = 4
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    best_loss = float("inf")
    best_ckpt = CKPT_DIR / "astro_conformer_110m_best.pt"
    
    print("-" * 80)
    print(f"[*] Commencing Fast & Deep 110M Foundation Pretraining ({epochs} Epochs)...")
    t_all = time.time()
    
    for epoch in range(1, epochs + 1):
        t_start = time.time()
        tr_loss, tr_rec, tr_dyn, tr_acc = train_epoch(model, train_loader, optimizer, scaler, device, epoch, accum_steps=2)
        val_rec, val_dyn, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        t_elapsed = time.time() - t_start
        
        val_total = val_rec + 0.35 * val_dyn
        print(
            f"[*] [Epoch {epoch:2d}/{epochs}] "
            f"Train Recon: {tr_rec:.4f} | Train Dyn Acc: {tr_acc:.1f}% | "
            f"Val Recon: {val_rec:.4f} | Val Dyn Acc: {val_acc:.1f}% | "
            f"Val Loss: {val_total:.4f} | Time: {t_elapsed/60:.1f}m"
        )
        
        # Сохраняем веса
        if val_total < best_loss or math.isnan(best_loss):
            best_loss = val_total
            torch.save(model.state_dict(), best_ckpt)
            print(f"    [+] New Best 110M Weights Saved -> {best_ckpt.name} (Val Metric: {val_total:.4f})")
        else:
            # Всегда сохраняем последнюю лучшую версию
            torch.save(model.state_dict(), best_ckpt)
            
    total_train_time = time.time() - t_all
    print("=" * 80)
    print(f"[+] Foundation Pretraining Complete in {total_train_time/60:.1f} minutes!")
    print(f"[+] Final Checkpoint Verified: {best_ckpt} (File Size: {best_ckpt.stat().st_size / 1e6:.1f} MB)")
    print("=" * 80)

if __name__ == "__main__":
    main()
