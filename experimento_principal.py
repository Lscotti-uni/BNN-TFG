"""
Experimento principal: Comparativa de modelos determinista, MC Dropout y BNN
sobre MNIST (in-distribution) y Fashion-MNIST (out-of-distribution).

Requisitos:
    pip install torch torchvision matplotlib seaborn scipy

Uso:
    python experimento_principal.py

Los resultados se guardan en la carpeta ./resultados/
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import entropy as scipy_entropy

# ─────────────────────────────────────────────
# Reproducibilidad
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")
if DEVICE.type == "cpu":
    print("⚠️  No se detecta GPU. En Colab: Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU")

os.makedirs("resultados", exist_ok=True)
os.makedirs("resultados/figuras", exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 1. DATOS
# ═══════════════════════════════════════════════════════════════

BATCH_SIZE = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST("./datos", train=True,  download=True, transform=transform)
mnist_test  = datasets.MNIST("./datos", train=False, download=True, transform=transform)
fmnist_test = datasets.FashionMNIST("./datos", train=False, download=True, transform=transform)

PIN = DEVICE.type == "cuda"

train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=PIN)
test_loader  = DataLoader(mnist_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=PIN)
ood_loader   = DataLoader(fmnist_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=PIN)

print(f"Train: {len(mnist_train)} | Test MNIST: {len(mnist_test)} | Test OOD: {len(fmnist_test)}")


# ═══════════════════════════════════════════════════════════════
# 2. ARQUITECTURAS
# ═══════════════════════════════════════════════════════════════
#
# Los tres modelos comparten exactamente la misma estructura CNN:
#
#   Conv2d(1,  16, kernel=5) → ReLU → MaxPool2d(2)   [28→12]
#   Conv2d(16, 32, kernel=5) → ReLU → MaxPool2d(2)   [12→4]
#   Flatten → 512
#   Linear(512, 128) → ReLU
#   Linear(128, 10)
#
# La única diferencia entre modelos está en cómo se tratan
# los pesos: deterministas, con dropout o distribuciones gaussianas.
# ═══════════════════════════════════════════════════════════════


# ── 2.0  Primitivas bayesianas ───────────────────────────────

def _softplus(rho: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.exp(rho))


def _kl_normal(mu_q: torch.Tensor, sigma_q: torch.Tensor,
               prior_sigma: float) -> torch.Tensor:
    return (
        np.log(prior_sigma) - torch.log(sigma_q)
        + (sigma_q ** 2 + mu_q ** 2) / (2.0 * prior_sigma ** 2)
        - 0.5
    ).sum()


class CapaBayesianaLineal(nn.Module):
    def __init__(self, in_f: int, out_f: int, prior_sigma: float):
        super().__init__()
        self.prior_sigma = prior_sigma
        self.w_mu  = nn.Parameter(torch.zeros(out_f, in_f))
        self.w_rho = nn.Parameter(torch.full((out_f, in_f), -3.0))
        self.b_mu  = nn.Parameter(torch.zeros(out_f))
        self.b_rho = nn.Parameter(torch.full((out_f,), -3.0))

    def forward(self, x: torch.Tensor):
        w_sigma = _softplus(self.w_rho)
        b_sigma = _softplus(self.b_rho)
        w = self.w_mu + w_sigma * torch.randn_like(w_sigma)
        b = self.b_mu + b_sigma * torch.randn_like(b_sigma)
        kl = _kl_normal(self.w_mu, w_sigma, self.prior_sigma) \
           + _kl_normal(self.b_mu, b_sigma, self.prior_sigma)
        return F.linear(x, w, b), kl


class CapaBayesianaConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 prior_sigma: float, stride: int = 1, padding: int = 0):
        super().__init__()
        self.prior_sigma = prior_sigma
        self.stride  = stride
        self.padding = padding
        shape = (out_ch, in_ch, kernel, kernel)
        self.w_mu  = nn.Parameter(torch.zeros(*shape))
        self.w_rho = nn.Parameter(torch.full(shape, -3.0))
        self.b_mu  = nn.Parameter(torch.zeros(out_ch))
        self.b_rho = nn.Parameter(torch.full((out_ch,), -3.0))

    def forward(self, x: torch.Tensor):
        w_sigma = _softplus(self.w_rho)
        b_sigma = _softplus(self.b_rho)
        w = self.w_mu + w_sigma * torch.randn_like(w_sigma)
        b = self.b_mu + b_sigma * torch.randn_like(b_sigma)
        kl = _kl_normal(self.w_mu, w_sigma, self.prior_sigma) \
           + _kl_normal(self.b_mu, b_sigma, self.prior_sigma)
        out = F.conv2d(x, w, b, stride=self.stride, padding=self.padding)
        return out, kl


# ── 2.1  Modelo determinista ─────────────────────────────────

class ModeloDeterminista(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.pool  = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1   = nn.Linear(512, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ── 2.2  Modelo MC Dropout ────────────────────────────────────

class ModeloMCDropout(nn.Module):
    def __init__(self, p_drop: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.pool  = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.drop  = nn.Dropout(p=p_drop)
        self.fc1   = nn.Linear(512, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

    def predecir_mc(self, x: torch.Tensor, T: int = 50):
        self.train()
        with torch.no_grad():
            probs = torch.stack(
                [F.softmax(self.forward(x), dim=1) for _ in range(T)], dim=0
            )
        self.eval()
        return probs.mean(dim=0), probs.var(dim=0)


# ── 2.3  BNN con Bayes by Backprop (CNN bayesiana) ───────────

class ModeloBNN(nn.Module):
    def __init__(self, prior_sigma: float = 1.0):
        super().__init__()
        ps = prior_sigma
        self.conv1 = CapaBayesianaConv2d(1,  16, kernel=5, prior_sigma=ps)
        self.pool  = nn.MaxPool2d(2)
        self.conv2 = CapaBayesianaConv2d(16, 32, kernel=5, prior_sigma=ps)
        self.fc1   = CapaBayesianaLineal(512, 128, prior_sigma=ps)
        self.fc2   = CapaBayesianaLineal(128,  10, prior_sigma=ps)

    def forward(self, x: torch.Tensor):
        kl = torch.tensor(0.0, device=x.device)
        out, kl_c = self.conv1(x);  kl = kl + kl_c
        out = self.pool(F.relu(out))
        out, kl_c = self.conv2(out); kl = kl + kl_c
        out = self.pool(F.relu(out))
        out = out.view(out.size(0), -1)
        out, kl_c = self.fc1(out);  kl = kl + kl_c
        out = F.relu(out)
        out, kl_c = self.fc2(out);  kl = kl + kl_c
        return out, kl

    def predecir_mc(self, x: torch.Tensor, T: int = 50):
        self.train()
        with torch.no_grad():
            probs = []
            for _ in range(T):
                logits, _ = self.forward(x)
                probs.append(F.softmax(logits, dim=1))
            probs = torch.stack(probs, dim=0)
        self.eval()
        return probs.mean(dim=0), probs.var(dim=0)


# ═══════════════════════════════════════════════════════════════
# 3. ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════

EPOCHS  = 50
LR_DET  = 1e-3   # determinista y MC Dropout
LR_BNN  = 1e-2   # BNN — warm-up + scheduler lo ajusta después
T_MC    = 50


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.mejor     = float("inf")
        self.contador  = 0

    def step(self, loss: float) -> bool:
        if loss < self.mejor - self.min_delta:
            self.mejor    = loss
            self.contador = 0
        else:
            self.contador += 1
        return self.contador >= self.patience


def entrenar_determinista(modelo: nn.Module, epochs: int = EPOCHS):
    opt       = torch.optim.Adam(modelo.parameters(), lr=LR_DET)
    criterion = nn.CrossEntropyLoss()
    historial = []
    es        = EarlyStopping(patience=5, min_delta=0.001)
    modelo.to(DEVICE)
    for epoch in range(1, epochs + 1):
        modelo.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(modelo(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(train_loader)
        historial.append(avg)
        if epoch % 5 == 0:
            print(f"  [Det] Época {epoch:3d}/{epochs}  loss={avg:.4f}")
        if es.step(avg):
            print(f"  [Det] Early stopping en época {epoch}")
            break
    return historial


def entrenar_mc_dropout(modelo: nn.Module, epochs: int = EPOCHS):
    return entrenar_determinista(modelo, epochs)


def entrenar_bnn(modelo: nn.Module, epochs: int = EPOCHS):
    """
    ELBO = E_q[log p(D|w)] - beta * KL(q||p) / N_batches

    KL warm-up: beta sube de 0 a 1 durante las primeras 20 épocas.
    ReduceLROnPlateau: a partir de la época 21 reduce el lr a la mitad
    si la pérdida no mejora en 3 épocas consecutivas (min lr = 1e-4).
    """
    opt       = torch.optim.Adam(modelo.parameters(), lr=LR_BNN)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode='min', factor=0.5, patience=3, min_lr=1e-4)
    N         = len(train_loader)
    historial = []
    es        = EarlyStopping(patience=10, min_delta=0.01)
    modelo.to(DEVICE)
    for epoch in range(1, epochs + 1):
        modelo.train()
        total_loss = 0.0
        kl_weight  = min(1.0, epoch / 20)
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, kl = modelo(x)
            nll  = F.cross_entropy(logits, y)
            loss = nll + kl_weight * kl / N
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / N
        historial.append(avg)
        if epoch > 20:
            scheduler.step(avg)
        if epoch % 5 == 0:
            lr_actual = opt.param_groups[0]['lr']
            print(f"  [BNN] Época {epoch:3d}/{epochs}  loss={avg:.4f}  kl_w={kl_weight:.2f}  lr={lr_actual:.5f}")
        if epoch <= 20:
            es = EarlyStopping(patience=10, min_delta=0.01)  # resetea durante warm-up
        elif es.step(avg):
            print(f"  [BNN] Early stopping en época {epoch}")
            break
    return historial


# ═══════════════════════════════════════════════════════════════
# 4. MÉTRICAS
# ═══════════════════════════════════════════════════════════════

def calcular_metricas_determinista(modelo: nn.Module, loader: DataLoader):
    modelo.eval()
    correcto, total = 0, 0
    entropias, confianzas = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            probs = F.softmax(modelo(x), dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            correcto += (preds == y.cpu().numpy()).sum()
            total    += len(y)
            for p in probs:
                entropias.append(scipy_entropy(p))
                confianzas.append(p.max())
    return {
        "accuracy":       correcto / total,
        "entropia_media": float(np.mean(entropias)),
        "entropia_std":   float(np.std(entropias)),
        "confianza_media":float(np.mean(confianzas)),
        "confianza_std":  float(np.std(confianzas)),
        "entropias":      entropias,
        "confianzas":     confianzas,
    }


def calcular_metricas_estocastico(modelo: nn.Module, loader: DataLoader, T: int = T_MC):
    modelo.eval()
    correcto, total = 0, 0
    entropias, varianzas, confianzas = [], [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        mean_p, var_p = modelo.predecir_mc(x, T=T)
        mean_p = mean_p.cpu().numpy()
        var_p  = var_p.cpu().numpy()
        preds  = mean_p.argmax(axis=1)
        correcto += (preds == y.cpu().numpy()).sum()
        total    += len(y)
        for p, v in zip(mean_p, var_p):
            entropias.append(scipy_entropy(p))
            varianzas.append(float(v.mean()))
            confianzas.append(float(p.max()))
    return {
        "accuracy":       correcto / total,
        "entropia_media": float(np.mean(entropias)),
        "entropia_std":   float(np.std(entropias)),
        "varianza_media": float(np.mean(varianzas)),
        "varianza_std":   float(np.std(varianzas)),
        "confianza_media":float(np.mean(confianzas)),
        "confianza_std":  float(np.std(confianzas)),
        "entropias":      entropias,
        "varianzas":      varianzas,
        "confianzas":     confianzas,
    }


def calcular_ece(confianzas: list, labels_correcto: list, n_bins: int = 10) -> float:
    confianzas = np.array(confianzas)
    correcto   = np.array(labels_correcto, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, N = 0.0, len(confianzas)
    for i in range(n_bins):
        mask = (confianzas >= bins[i]) & (confianzas < bins[i+1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / N * abs(correcto[mask].mean() - confianzas[mask].mean())
    return float(ece)


def calcular_ece_completo(modelo, loader, estocastico: bool = False, T: int = T_MC):
    modelo.eval()
    confianzas_list, aciertos_list = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if estocastico:
            mean_p, _ = modelo.predecir_mc(x, T=T)
            probs = mean_p.cpu().numpy()
        else:
            with torch.no_grad():
                probs = F.softmax(modelo(x), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for p, pred, label in zip(probs, preds, y.cpu().numpy()):
            confianzas_list.append(p.max())
            aciertos_list.append(int(pred == label))
    return calcular_ece(confianzas_list, aciertos_list)


def calcular_proper_scores(modelo, loader, estocastico: bool = False, T: int = T_MC):
    modelo.eval()
    log_scores, brier_scores = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if estocastico:
            mean_p, _ = modelo.predecir_mc(x, T=T)
            probs = mean_p.cpu().numpy()
        else:
            with torch.no_grad():
                probs = F.softmax(modelo(x), dim=1).cpu().numpy()
        for p_vec, label in zip(probs, y.cpu().numpy()):
            log_scores.append(float(np.log(p_vec[label] + 1e-12)))
            onehot = np.zeros(10); onehot[label] = 1.0
            brier_scores.append(float(np.mean((p_vec - onehot) ** 2)))
    return {
        "log_score":   float(np.mean(log_scores)),
        "brier_score": float(np.mean(brier_scores)),
    }


# ═══════════════════════════════════════════════════════════════
# 5. SISTEMA DE RECHAZO AUTOMÁTICO
# ═══════════════════════════════════════════════════════════════

def evaluar_con_rechazo(modelo, loader, umbral: float,
                        estocastico: bool = False, T: int = T_MC,
                        metrica: str = "entropia"):
    modelo.eval()
    aceptadas_correctas, aceptadas_total, rechazadas = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if estocastico:
            mean_p, var_p = modelo.predecir_mc(x, T=T)
            probs = mean_p.cpu().numpy()
            var_p = var_p.cpu().numpy()
        else:
            with torch.no_grad():
                probs = F.softmax(modelo(x), dim=1).cpu().numpy()
            var_p = np.zeros_like(probs)
        preds = probs.argmax(axis=1)
        for p, v, pred, label in zip(probs, var_p, preds, y.cpu().numpy()):
            score = scipy_entropy(p) if metrica == "entropia" else float(v.mean())
            if score > umbral:
                rechazadas += 1
            else:
                aceptadas_total += 1
                if pred == label:
                    aceptadas_correctas += 1
    total = aceptadas_total + rechazadas
    return {
        "accuracy_aceptadas": aceptadas_correctas / aceptadas_total if aceptadas_total > 0 else 0.0,
        "tasa_rechazo":       rechazadas / total,
        "n_aceptadas":        aceptadas_total,
    }


def curva_rechazo(modelo, loader_id, loader_ood,
                  estocastico: bool = False, T: int = T_MC, n_umbrales: int = 20):
    modelo.eval()

    def _entropias(loader):
        ents = []
        for x, _ in loader:
            x = x.to(DEVICE)
            if estocastico:
                mean_p, _ = modelo.predecir_mc(x, T=T)
                probs = mean_p.cpu().numpy()
            else:
                with torch.no_grad():
                    probs = F.softmax(modelo(x), dim=1).cpu().numpy()
            for p in probs:
                ents.append(scipy_entropy(p))
        return ents

    ents_id  = _entropias(loader_id)
    ents_ood = _entropias(loader_ood)
    umbrales = np.linspace(0, np.log(10), n_umbrales)
    tasa_id  = [np.mean(np.array(ents_id)  > u) for u in umbrales]
    tasa_ood = [np.mean(np.array(ents_ood) > u) for u in umbrales]
    return umbrales, tasa_id, tasa_ood, ents_id, ents_ood


# ═══════════════════════════════════════════════════════════════
# 6. VISUALIZACIONES
# ═══════════════════════════════════════════════════════════════

def plot_histogramas_entropia(resultados: dict, nombre_fig: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    nombres = ["Determinista", "MC Dropout", "BNN"]
    keys    = ["det", "mc", "bnn"]
    colores = {"id": "#4C72B0", "ood": "#DD8452"}
    for ax, nombre, key in zip(axes, nombres, keys):
        ents_id  = resultados[key]["mnist"]["entropias"]
        ents_ood = resultados[key]["fmnist"]["entropias"]
        ax.hist(ents_id,  bins=50, alpha=0.6, color=colores["id"],  label="MNIST (ID)",        density=True)
        ax.hist(ents_ood, bins=50, alpha=0.6, color=colores["ood"], label="FashionMNIST (OOD)", density=True)
        ax.set_title(nombre, fontsize=13, fontweight="bold")
        ax.set_xlabel("Entropía predictiva $H$", fontsize=11)
        ax.set_ylabel("Densidad", fontsize=11)
        ax.axvline(np.mean(ents_id),  color=colores["id"],  linestyle="--", linewidth=1.5)
        ax.axvline(np.mean(ents_ood), color=colores["ood"], linestyle="--", linewidth=1.5)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.suptitle("Distribución de la entropía predictiva: ID vs OOD", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"resultados/figuras/{nombre_fig}.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"resultados/figuras/{nombre_fig}.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → Figura guardada: {nombre_fig}")


def plot_curvas_rechazo(curvas: dict, nombre_fig: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    nombres = ["Determinista", "MC Dropout", "BNN"]
    keys    = ["det", "mc", "bnn"]
    colores = ["#4C72B0", "#55A868", "#C44E52"]
    ax = axes[0]
    for nombre, key, col in zip(nombres, keys, colores):
        u, _, tasa_ood, _, _ = curvas[key]
        ax.plot(u, tasa_ood, label=nombre, color=col, linewidth=2)
    ax.set_xlabel("Umbral de entropía $\\tau$", fontsize=11)
    ax.set_ylabel("Fracción rechazada en OOD", fontsize=11)
    ax.set_title("Detección de OOD mediante rechazo", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    for nombre, key, col in zip(nombres, keys, colores):
        u, tasa_id, _, _, _ = curvas[key]
        ax.plot(u, tasa_id, label=nombre, color=col, linewidth=2)
    ax.set_xlabel("Umbral de entropía $\\tau$", fontsize=11)
    ax.set_ylabel("Fracción rechazada en ID", fontsize=11)
    ax.set_title("Coste del rechazo en datos ID", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle("Curvas de rechazo automático", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"resultados/figuras/{nombre_fig}.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"resultados/figuras/{nombre_fig}.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → Figura guardada: {nombre_fig}")


def plot_reliability_diagram(modelo, loader, estocastico: bool,
                              nombre_modelo: str, nombre_fig: str,
                              T: int = T_MC, n_bins: int = 10):
    confianzas_list, aciertos_list = [], []
    modelo.eval()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if estocastico:
            mean_p, _ = modelo.predecir_mc(x, T=T)
            probs = mean_p.cpu().numpy()
        else:
            with torch.no_grad():
                probs = F.softmax(modelo(x), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for p, pred, label in zip(probs, preds, y.cpu().numpy()):
            confianzas_list.append(p.max())
            aciertos_list.append(int(pred == label))
    confianzas_arr = np.array(confianzas_list)
    aciertos_arr   = np.array(aciertos_list, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    acc_bins, conf_bins = [], []
    for i in range(n_bins):
        mask = (confianzas_arr >= bins[i]) & (confianzas_arr < bins[i+1])
        if mask.sum() == 0:
            continue
        acc_bins.append(aciertos_arr[mask].mean())
        conf_bins.append(confianzas_arr[mask].mean())
    ece = calcular_ece(confianzas_list, aciertos_list, n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Calibración perfecta")
    ax.bar(conf_bins, acc_bins, width=0.08, alpha=0.6,
           color="#4C72B0", edgecolor="navy", label=f"ECE = {ece:.4f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confianza media del modelo", fontsize=11)
    ax.set_ylabel("Precisión observada", fontsize=11)
    ax.set_title(f"Reliability Diagram – {nombre_modelo}", fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"resultados/figuras/{nombre_fig}.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"resultados/figuras/{nombre_fig}.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → Figura guardada: {nombre_fig}")


def plot_ejemplos_representativos(modelos: dict, nombre_fig: str, T: int = T_MC):
    mnist_iter  = iter(DataLoader(mnist_test,  batch_size=256, shuffle=True))
    fmnist_iter = iter(DataLoader(fmnist_test, batch_size=64,  shuffle=True))
    x_id, y_id = next(mnist_iter)
    x_ood, _   = next(fmnist_iter)
    det = modelos["det"]; det.eval()
    with torch.no_grad():
        preds = det(x_id.to(DEVICE)).argmax(dim=1).cpu()
    correcto_idx   = (preds == y_id).nonzero(as_tuple=True)[0][0].item()
    incorrectos    = (preds != y_id).nonzero(as_tuple=True)[0]
    incorrecto_idx = correcto_idx if len(incorrectos) == 0 else incorrectos[0].item()
    casos = [
        (x_id[correcto_idx],   y_id[correcto_idx].item(),  "ID correcto",   True),
        (x_id[incorrecto_idx], y_id[incorrecto_idx].item(),"ID incorrecto", True),
        (x_ood[0],             -1,                          "OOD",           False),
    ]
    nombres_modelo = ["Determinista", "MC Dropout", "BNN"]
    keys_modelo    = ["det", "mc", "bnn"]
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 4, figure=fig, wspace=0.35, hspace=0.5)
    for fila, (img, label, titulo, es_id) in enumerate(casos):
        ax_img = fig.add_subplot(gs[fila, 0])
        ax_img.imshow(img.squeeze(), cmap="gray")
        ax_img.set_title(f"{titulo}\n{'Clase real: ' + str(label) if es_id else 'OOD'}", fontsize=10)
        ax_img.axis("off")
        img_batch = img.unsqueeze(0).to(DEVICE)
        for col, (key, nombre_m) in enumerate(zip(keys_modelo, nombres_modelo), start=1):
            modelo = modelos[key]; modelo.eval()
            if key == "det":
                with torch.no_grad():
                    p = F.softmax(modelo(img_batch), dim=1).cpu().numpy()[0]
            else:
                mean_p, _ = modelo.predecir_mc(img_batch, T=T)
                p = mean_p.cpu().numpy()[0]
            ent = scipy_entropy(p)
            ax  = fig.add_subplot(gs[fila, col])
            colores_barras = ["#DD8452" if i == p.argmax() else "#4C72B0" for i in range(10)]
            ax.bar(range(10), p, color=colores_barras, edgecolor="none")
            ax.set_xticks(range(10)); ax.set_xticklabels([str(i) for i in range(10)], fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(f"{nombre_m}\nH={ent:.3f}", fontsize=9)
            ax.axhline(0.1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            if col == 1:
                ax.set_ylabel("Probabilidad", fontsize=8)
    fig.suptitle("Distribuciones predictivas: casos representativos", fontsize=13)
    plt.savefig(f"resultados/figuras/{nombre_fig}.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"resultados/figuras/{nombre_fig}.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → Figura guardada: {nombre_fig}")


def plot_historial_entrenamiento(historiales: dict, nombre_fig: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    colores = {"det": "#4C72B0", "mc": "#55A868", "bnn": "#C44E52"}
    labels  = {"det": "Determinista", "mc": "MC Dropout", "bnn": "BNN"}
    for key, hist in historiales.items():
        ax.plot(range(1, len(hist)+1), hist, label=labels[key], color=colores[key], linewidth=2)
    ax.set_xlabel("Época", fontsize=11)
    ax.set_ylabel("Pérdida media (train)", fontsize=11)
    ax.set_title("Curvas de entrenamiento", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"resultados/figuras/{nombre_fig}.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"resultados/figuras/{nombre_fig}.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → Figura guardada: {nombre_fig}")


def plot_tabla_comparativa(tabla: dict, scores: dict, nombre_fig: str):
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.axis("off")
    filas = []
    cols  = ["Modelo", "Acc. MNIST", "H MNIST", "H FashionMNIST", "ECE MNIST", "Log-score", "Brier"]
    for key, datos in tabla.items():
        ps = scores[key]
        filas.append([
            datos["nombre"],
            f"{datos['acc_id']:.4f}",
            f"{datos['H_id']:.4f}",
            f"{datos['H_ood']:.4f}",
            f"{datos['ece']:.4f}",
            f"{ps['log_score']:.4f}",
            f"{ps['brier_score']:.4f}",
        ])
    tbl = ax.table(cellText=filas, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2E75B6")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    plt.title("Tabla comparativa de métricas", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(f"resultados/figuras/{nombre_fig}.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"resultados/figuras/{nombre_fig}.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → Figura guardada: {nombre_fig}")


def plot_selective_accuracy(modelos_dict: dict, loader, nombre_fig: str, T: int = T_MC):
    fig, ax = plt.subplots(figsize=(8, 5))
    colores = {"det": "#4C72B0", "mc": "#55A868", "bnn": "#C44E52"}
    labels  = {"det": "Determinista", "mc": "MC Dropout", "bnn": "BNN"}
    for key, modelo in modelos_dict.items():
        modelo.eval()
        entropias, correctos = [], []
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if key == "det":
                with torch.no_grad():
                    probs = F.softmax(modelo(x), dim=1).cpu().numpy()
            else:
                mean_p, _ = modelo.predecir_mc(x, T=T)
                probs = mean_p.cpu().numpy()
            preds = probs.argmax(axis=1)
            for p_vec, pred, label in zip(probs, preds, y.cpu().numpy()):
                entropias.append(scipy_entropy(p_vec))
                correctos.append(int(pred == label))
        entropias = np.array(entropias)
        correctos = np.array(correctos, dtype=float)
        orden         = np.argsort(entropias)
        correctos_ord = correctos[orden]
        n = len(correctos_ord)
        fracciones_rechazo = np.linspace(0, 0.99, 100)
        sel_acc = []
        for frac in fracciones_rechazo:
            n_aceptados = int(n * (1 - frac))
            if n_aceptados == 0:
                sel_acc.append(np.nan)
            else:
                sel_acc.append(correctos_ord[:n_aceptados].mean())
        ax.plot(fracciones_rechazo * 100, sel_acc,
                label=labels[key], color=colores[key], linewidth=2)
    ax.set_xlabel("Fracción rechazada (%)", fontsize=11)
    ax.set_ylabel("Selective accuracy", fontsize=11)
    ax.set_title("Selective accuracy en función del umbral de rechazo (MNIST)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.9, 1.005)
    plt.tight_layout()
    plt.savefig(f"resultados/figuras/{nombre_fig}.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"resultados/figuras/{nombre_fig}.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → Figura guardada: {nombre_fig}")


# ═══════════════════════════════════════════════════════════════
# 7. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  EXPERIMENTO: Determinista vs MC Dropout vs BNN")
    print("="*60 + "\n")

    det = ModeloDeterminista()
    mc  = ModeloMCDropout(p_drop=0.3)
    bnn = ModeloBNN(prior_sigma=0.5)

    print("▶ Entrenando modelo DETERMINISTA ...")
    hist_det = entrenar_determinista(det)

    print("\n▶ Entrenando modelo MC DROPOUT ...")
    hist_mc  = entrenar_mc_dropout(mc)

    print("\n▶ Entrenando modelo BNN (Bayes by Backprop) ...")
    hist_bnn = entrenar_bnn(bnn)

    historiales = {"det": hist_det, "mc": hist_mc, "bnn": hist_bnn}
    plot_historial_entrenamiento(historiales, "00_historial_entrenamiento")

    torch.save(det.state_dict(), "resultados/det.pt")
    torch.save(mc.state_dict(),  "resultados/mc.pt")
    torch.save(bnn.state_dict(), "resultados/bnn.pt")
    print("\n✓ Pesos guardados en resultados/")

    print("\n▶ Calculando métricas ...")
    det.eval(); mc.eval(); bnn.eval()

    r_det_id  = calcular_metricas_determinista(det, test_loader)
    r_det_ood = calcular_metricas_determinista(det, ood_loader)
    r_mc_id   = calcular_metricas_estocastico(mc,  test_loader)
    r_mc_ood  = calcular_metricas_estocastico(mc,  ood_loader)
    r_bnn_id  = calcular_metricas_estocastico(bnn, test_loader)
    r_bnn_ood = calcular_metricas_estocastico(bnn, ood_loader)

    ece_det = calcular_ece_completo(det, test_loader, estocastico=False)
    ece_mc  = calcular_ece_completo(mc,  test_loader, estocastico=True)
    ece_bnn = calcular_ece_completo(bnn, test_loader, estocastico=True)

    ps_det  = calcular_proper_scores(det, test_loader, estocastico=False)
    ps_mc   = calcular_proper_scores(mc,  test_loader, estocastico=True)
    ps_bnn  = calcular_proper_scores(bnn, test_loader, estocastico=True)
    scores  = {"det": ps_det, "mc": ps_mc, "bnn": ps_bnn}

    tabla = {
        "det": {"nombre": "Determinista", "acc_id": r_det_id["accuracy"],
                "H_id": r_det_id["entropia_media"], "H_ood": r_det_ood["entropia_media"],
                "ece": ece_det},
        "mc":  {"nombre": "MC Dropout",   "acc_id": r_mc_id["accuracy"],
                "H_id": r_mc_id["entropia_media"],  "H_ood": r_mc_ood["entropia_media"],
                "ece": ece_mc},
        "bnn": {"nombre": "BNN",           "acc_id": r_bnn_id["accuracy"],
                "H_id": r_bnn_id["entropia_media"], "H_ood": r_bnn_ood["entropia_media"],
                "ece": ece_bnn},
    }

    print("\n─── RESULTADOS ───────────────────────────────")
    for key, d in tabla.items():
        ps = scores[key]
        print(f"  {d['nombre']:15s}  Acc={d['acc_id']:.4f}  "
              f"H_id={d['H_id']:.4f}  H_ood={d['H_ood']:.4f}  "
              f"ECE={d['ece']:.4f}  LogScore={ps['log_score']:.4f}  "
              f"Brier={ps['brier_score']:.4f}")

    def limpiar(d):
        return {k: v for k, v in d.items() if not isinstance(v, list)}

    json_limpio = {
        "det": {"mnist": limpiar(r_det_id), "fmnist": limpiar(r_det_ood),
                "ece": ece_det, "proper_scores": ps_det},
        "mc":  {"mnist": limpiar(r_mc_id),  "fmnist": limpiar(r_mc_ood),
                "ece": ece_mc,  "proper_scores": ps_mc},
        "bnn": {"mnist": limpiar(r_bnn_id), "fmnist": limpiar(r_bnn_ood),
                "ece": ece_bnn, "proper_scores": ps_bnn},
    }
    with open("resultados/metricas.json", "w") as f:
        json.dump(json_limpio, f, indent=2)
    print("✓ Métricas guardadas en resultados/metricas.json")

    print("\n▶ Generando figuras ...")
    plot_tabla_comparativa(tabla, scores, "01_tabla_comparativa")

    resultados_completos = {
        "det": {"mnist": r_det_id, "fmnist": r_det_ood},
        "mc":  {"mnist": r_mc_id,  "fmnist": r_mc_ood},
        "bnn": {"mnist": r_bnn_id, "fmnist": r_bnn_ood},
    }
    plot_histogramas_entropia(resultados_completos, "02_histogramas_entropia")

    plot_reliability_diagram(det, test_loader, False, "Determinista", "03a_reliability_det")
    plot_reliability_diagram(mc,  test_loader, True,  "MC Dropout",   "03b_reliability_mc")
    plot_reliability_diagram(bnn, test_loader, True,  "BNN",          "03c_reliability_bnn")

    print("  Calculando curvas de rechazo (puede tardar unos minutos) ...")
    curvas = {
        "det": curva_rechazo(det, test_loader, ood_loader, estocastico=False),
        "mc":  curva_rechazo(mc,  test_loader, ood_loader, estocastico=True),
        "bnn": curva_rechazo(bnn, test_loader, ood_loader, estocastico=True),
    }
    plot_curvas_rechazo(curvas, "04_curvas_rechazo")

    print("\n▶ Evaluando sistema de rechazo (umbral=0.5) ...")
    from torch.utils.data import ConcatDataset
    loader_mixto = DataLoader(
        ConcatDataset([mnist_test, fmnist_test]),
        batch_size=BATCH_SIZE, shuffle=False
    )
    for key, nombre in [("det", "Determinista"), ("mc", "MC Dropout"), ("bnn", "BNN")]:
        modelo  = {"det": det, "mc": mc, "bnn": bnn}[key]
        es_esto = key != "det"
        res = evaluar_con_rechazo(modelo, loader_mixto, umbral=0.5, estocastico=es_esto)
        print(f"  {nombre:15s}  Acc_aceptadas={res['accuracy_aceptadas']:.4f}  "
              f"Tasa_rechazo={res['tasa_rechazo']:.4f}")

    modelos_dict = {"det": det, "mc": mc, "bnn": bnn}

    print("  Calculando selective accuracy ...")
    plot_selective_accuracy(modelos_dict, test_loader, "06_selective_accuracy")

    plot_ejemplos_representativos(modelos_dict, "05_ejemplos_representativos")

    print("\n" + "="*60)
    print("  EXPERIMENTO COMPLETADO")
    print(f"  Figuras en: resultados/figuras/")
    print(f"  Métricas:   resultados/metricas.json")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
