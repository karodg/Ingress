import math
import torch
import numpy as np
from torch.utils.data import Dataset

PHENOMENA_MAP = {
    0: "Standard Stellar / Quiescent Noise",
    1: "Exoplanetary Transit (Keplerian Multi-Planet)",
    2: "Disintegrating Exocomet (Asymmetric Dust Tail)",
    3: "Gravitational Microlensing (Paczynski Curve)",
    4: "Stellar Megaflare (Coronal Mass Ejection)",
    5: "Circumstellar Dust Swarm (Tabby's Star Type)",
    6: "Eclipsing Binary System (Primary & Secondary Eclipses)"
}

class Astro100MDataset(Dataset):
    """
    Масштабированный астрофизический датасет для 110M Foundation Model.
    
    Генерирует астрофизические световые кривые с реалистичной физикой:
      - 7 категорий явлений (от экзопланет до гравитационного микролинзирования)
      - Астеросейсмологические пульсации p-мод
      - Дифференциальное вращение и пятнистость звезд
      - Случайный инструментальный шум TESS/Kepler
    """
    def __init__(self, size=15000, seq_len=2048):
        self.size = size
        self.seq_len = seq_len
        self.t_base = np.arange(seq_len, dtype=np.float32)

    def __len__(self):
        return self.size

    def _generate_cosmic_system(self, t):
        # 1. Астеросейсмологический фон и звездные пятна
        rot_p = np.random.uniform(40.0, 600.0)
        spot_a = np.random.uniform(0.001, 0.03)
        flux = 1.0 + spot_a * np.sin(2 * np.pi * t / rot_p)
        flux += 0.5 * spot_a * np.sin(4 * np.pi * t / rot_p + np.random.uniform(0, math.pi))
        
        # p-моды акустических пульсаций
        p_per = np.random.uniform(4.0, 25.0)
        p_amp = np.random.uniform(0.0001, 0.002)
        flux += p_amp * np.sin(2 * np.pi * t / p_per)
        
        # 2. Выбор космического феномена (0..6)
        # 0: standard, 1: exoplanet, 2: disintegrating, 3: microlensing, 4: flare, 5: tabby_dust, 6: binary
        phenom_idx = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=[0.20, 0.30, 0.10, 0.10, 0.10, 0.10, 0.10])
        
        if phenom_idx == 1:
            # Экзопланетные транзиты (от 1 до 3 планет)
            n_planets = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            for _ in range(n_planets):
                period = np.random.uniform(80.0, 700.0)
                t0 = np.random.uniform(0.0, period)
                dur = np.random.uniform(12.0, 50.0)
                depth = np.random.uniform(0.0002, 0.015)
                phase = (t - t0) % period
                in_tr = (phase < dur)
                t_rel = (phase[in_tr] - dur / 2.0) / (dur / 2.0)
                flux[in_tr] *= (1.0 - depth * (1.0 - 0.35 * (t_rel**2)))
                
        elif phenom_idx == 2:
            # Распадающаяся экзокомета с асимметричным хвостом
            period = np.random.uniform(120.0, 350.0)
            t0 = np.random.uniform(0.0, period)
            dur = np.random.uniform(20.0, 60.0)
            depth = np.random.uniform(0.002, 0.02)
            phase = (t - t0) % period
            in_tr = (phase < dur)
            t_in = phase[in_tr]
            shape = np.where(t_in < 0.25 * dur, t_in / (0.25 * dur), np.exp(-(t_in - 0.25 * dur) / (0.35 * dur)))
            flux[in_tr] *= (1.0 - depth * shape)
            
        elif phenom_idx == 3:
            # Гравитационное микролинзирование (Paczyński)
            t0 = np.random.uniform(300.0, 1700.0)
            tE = np.random.uniform(40.0, 220.0)
            u0 = np.random.uniform(0.05, 0.45)
            u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
            amp = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
            flux *= amp
            
        elif phenom_idx == 4:
            # Звездная мегавспышка
            for _ in range(np.random.randint(1, 3)):
                t_pk = np.random.uniform(150.0, 1850.0)
                f_amp = np.random.uniform(0.01, 0.20)
                tau = np.random.uniform(8.0, 35.0)
                in_fl = (t >= t_pk)
                flux[in_fl] += f_amp * np.exp(-(t[in_fl] - t_pk) / tau)
                
        elif phenom_idx == 5:
            # Околозвездное облако пыли (Звезда Табби)
            for _ in range(np.random.randint(2, 6)):
                t_c = np.random.uniform(200.0, 1800.0)
                w = np.random.uniform(25.0, 100.0)
                d = np.random.uniform(0.03, 0.22)
                flux *= (1.0 - d * np.exp(-((np.abs(t - t_c) / w)**2)))
                
        elif phenom_idx == 6:
            # Затменная двойная система (главный и вторичный минимум)
            p_bin = np.random.uniform(100.0, 400.0)
            t0 = np.random.uniform(0.0, p_bin)
            dur = np.random.uniform(15.0, 40.0)
            d1 = np.random.uniform(0.03, 0.15)
            d2 = d1 * np.random.uniform(0.3, 0.8)
            in_tr1 = ((t - t0) % p_bin) < dur
            flux[in_tr1] *= (1.0 - d1)
            in_tr2 = ((t - t0 - p_bin * 0.5) % p_bin) < dur
            flux[in_tr2] *= (1.0 - d2)
            
        # 3. Инструментальный фотометрический шум
        noise_std = np.random.uniform(1e-4, 5e-4)
        noise = np.random.normal(0, noise_std, size=len(t))
        flux_raw = flux + noise
        
        # Нормализация
        med = np.median(flux_raw)
        std = np.std(flux_raw) + 1e-6
        x = (flux_raw - med) / std
        
        return x, phenom_idx

    def __getitem__(self, idx):
        x, label = self._generate_cosmic_system(self.t_base)
        return torch.from_numpy(x).float(), torch.tensor(label, dtype=torch.long)
