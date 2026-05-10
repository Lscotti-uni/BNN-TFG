"""
grid_search_bnn.py
==================
Grid search sobre learning_rate y prior_sigma para la BNN en PyTorch.
Objetivo: encontrar la combinación que minimiza ECE y maximiza accuracy,
es decir, la que hace que la BNN converja mejor.

Combinaciones: 3 lr × 3 prior_sigma = 9 experimentos.
Guarda resultados en resultados_grid/resultados.json
y una figura resumen en resultados_grid/resumen_convergencia.png

Uso en Colab:
    !python grid_search_bnn.py

Requisitos:
    pip install torch torchvision matplotlib scipy
"""

import os
import json
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy

# ─────────────────────────────────────────────
# Dispositivo
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")
if DEVICE.type == "cpu":
    print("⚠️  Sin GPU. En Colab: Entorno de ejecución → Cambiar tipo → GPU")

os.makedirs("resultados_grid", exist_ok=True)
os.makedirs("resultados_grid/figuras", exist_ok=True)

# ─────────────────────────────────────────────
# Grid de hiperparámetros
# ─────────────────────────────────────────────
LEARNING_RATES = [1e-4, 1e-2, 0.1]
PRIOR_SIGMAS   = [0.1, 0.5, 1.0]
EPOCHS         = 50
BATCH_SIZE     = 128
T_MC           = 50
SEED           = 42

# ─────────────────────────────────────────────
# Datos
# ─────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

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


# ═══════════════════════════════════════════════════════════════
# ARQUITECTURA BNN (igual que en experimento_principal.py)
# ═══════════════════════════════════════════════════════════════

def _softplus(rho):
    return torch.log1p(torch.exp(rho))

def _kl_normal(mu_q, sigma_q, prior_sigma):
    return (
        np.log(prior_sigma) - torch.log(sigma_q)
        + (sigma_q**2 + mu_q**2) / (2.0 * prior_sigma**2)
        - 0.5
    ).sum()

class CapaBayesianaLineal(nn.Module):
    def __init__(self, in_f, out_f, prior_sigma):
        super().__init__()
        self.prior_sigma = prior_sigma
        self.w_mu  = nn.Parameter(torch.zeros(out_f, in_f))
        self.w_rho = nn.Parameter(torch.full((out_f, in_f), -3.0))
        self.b_mu  = nn.Parameter(torch.zeros(out_f))
        self.b_rho = nn.Parameter(torch.full((out_f,), -3.0))

    def forward(self, x):
        w_s = _softplus(self.w_rho)
        b_s = _softplus(self.b_rho)
        w   = self.w_mu + w_s * torch.randn_like(w_s)
        b   = self.b_mu + b_s * torch.randn_like(b_s)
        kl  = _kl_normal(self.w_mu, w_s, self.prior_sigma) \
            + _kl_normal(self.b_mu, b_s, self.prior_sigma)
        return F.linear(x, w, b), kl

class CapaBayesianaConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, prior_sigma):
        super().__init__()
        self.prior_sigma = prior_sigma
        shape = (out_ch, in_ch, kernel, kernel)
        self.w_mu  = nn.Parameter(torch.zeros(*shape))
        self.w_rho = nn.Parameter(torch.full(shape, -3.0))
        self.b_mu  = nn.Parameter(torch.zeros(out_ch))
        self.b_rho = nn.Parameter(torch.full((out_ch,), -3.0))

    def forward(self, x):
        w_s = _softplus(self.w_rho)
        b_s = _softplus(self.b_rho)
        w   = self.w_mu + w_s * torch.randn_like(w_s)
        b   = self.b_mu + b_s * torch.randn_like(b_s)
        kl  = _kl_normal(self.w_mu, w_s, self.prior_sigma) \
            + _kl_normal(self.b_mu, b_s, self.prior_sigma)
        return F.conv2d(x, w, b), kl

class ModeloBNN(nn.Module):
    def __init__(self, prior_sigma=1.0):
        super().__init__()
        ps = prior_sigma
        self.conv1 = CapaBayesianaConv2d(1,  16, 5, ps)
        self.pool  = nn.MaxPool2d(2)
        self.conv2 = CapaBayesianaConv2d(16, 32, 5, ps)
        self.fc1   = CapaBayesianaLineal(512, 128, ps)
        self.fc2   = CapaBayesianaLineal(128,  10, ps)

    def forward(self, x):
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

    def predecir_mc(self, x, T=T_MC):
        self.train()
        with torch.no_grad():
            probs = torch.stack(
                [F.softmax(self.forward(x)[0], dim=1) for _ in range(T)], dim=0
            )
        self.eval()
        return probs.mean(dim=0), probs.var(dim=0)


# ═══════════════════════════════════════════════════════════════
# ENTRENAMIENTO CON EARLY STOPPING
# ═══════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.01):
        self.patience  = patience
        self.min_delta = min_delta
        self.mejor     = float("inf")
        self.contador  = 0

    def step(self, loss):
        if loss < self.mejor - self.min_delta:
            self.mejor    = loss
            self.contador = 0
        else:
            self.contador += 1
        return self.contador >= self.patience


def entrenar(modelo, lr, epochs=EPOCHS):
    opt       = torch.optim.Adam(modelo.parameters(), lr=lr)
    N         = len(train_loader)
    historial = []
    es        = EarlyStopping(patience=10, min_delta=0.01)
    modelo.to(DEVICE)

    for epoch in range(1, epochs + 1):
        modelo.train()
        total = 0.0
        kl_w  = min(1.0, epoch / 20)
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, kl = modelo(x)
            loss = F.cross_entropy(logits, y) + kl_w * kl / N
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / N
        historial.append(avg)
        if epoch % 10 == 0:
            print(f"    Época {epoch:3d}  loss={avg:.4f}  kl_w={kl_w:.2f}")
        if es.step(avg):
            print(f"    Early stopping en época {epoch}  loss={avg:.4f}")
            break

    return historial


# ═══════════════════════════════════════════════════════════════
# MÉTRICAS
# ═══════════════════════════════════════════════════════════════

def calcular_metricas(modelo, loader):
    modelo.eval()
    correcto, total = 0, 0
    entropias, log_scores, brier_scores = [], [], []
    confs, aciertos = [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        mean_p, _ = modelo.predecir_mc(x)
        probs  = mean_p.cpu().numpy()
        preds  = probs.argmax(axis=1)
        labels = y.cpu().numpy()
        correcto += (preds == labels).sum()
        total    += len(labels)
        for p, pred, label in zip(probs, preds, labels):
            entropias.append(float(scipy_entropy(p)))
            log_scores.append(float(np.log(p[label] + 1e-12)))
            onehot = np.zeros(10); onehot[label] = 1.0
            brier_scores.append(float(np.mean((p - onehot)**2)))
            confs.append(float(p.max()))
            aciertos.append(int(pred == label))

    # ECE
    confs_arr    = np.array(confs)
    aciertos_arr = np.array(aciertos, dtype=float)
    bins = np.linspace(0, 1, 11)
    ece, N = 0.0, len(confs_arr)
    for i in range(10):
        mask = (confs_arr >= bins[i]) & (confs_arr < bins[i+1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / N * abs(aciertos_arr[mask].mean() - confs_arr[mask].mean())

    return {
        "accuracy":       correcto / total,
        "entropia_media": float(np.mean(entropias)),
        "log_score":      float(np.mean(log_scores)),
        "brier_score":    float(np.mean(brier_scores)),
        "ece":            float(ece),
        "entropias":      entropias,
    }


# ═══════════════════════════════════════════════════════════════
# FIGURA POR COMBINACIÓN
# ═══════════════════════════════════════════════════════════════

def plot_combinacion(historial, m_id, m_ood, lr, ps, nombre):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"BNN — lr={lr}  prior_sigma={ps}", fontsize=13)

    # Curva de entrenamiento
    axes[0].plot(historial, color="#C44E52", linewidth=2)
    axes[0].set_xlabel("Época"); axes[0].set_ylabel("ELBO loss")
    axes[0].set_title("Convergencia"); axes[0].grid(alpha=0.3)
    # Marcar si ha convergido (plateau visible)
    if len(historial) < 50:
        axes[0].set_title(f"Convergencia (paró en época {len(historial)})", fontsize=10)

    # Histograma entropía ID vs OOD
    axes[1].hist(m_id["entropias"],  bins=40, alpha=0.6, color="#4C72B0",
                 label=f"MNIST (ID)  H={m_id['entropia_media']:.3f}",  density=True)
    axes[1].hist(m_ood["entropias"], bins=40, alpha=0.6, color="#DD8452",
                 label=f"OOD  H={m_ood['entropia_media']:.3f}", density=True)
    axes[1].set_xlabel("Entropía predictiva"); axes[1].set_ylabel("Densidad")
    axes[1].set_title("Entropía ID vs OOD")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    # Tabla de métricas — las dos clave resaltadas
    axes[2].axis("off")
    datos = [
        ["Accuracy",   f"{m_id['accuracy']:.4f}",    "↑"],
        ["ECE",        f"{m_id['ece']:.4f}",          "↓"],
        ["H_id",       f"{m_id['entropia_media']:.4f}", "↓"],
        ["H_ood",      f"{m_ood['entropia_media']:.4f}", "↑"],
        ["Log-score",  f"{m_id['log_score']:.4f}",   "↑"],
        ["Brier",      f"{m_id['brier_score']:.4f}",  "↓"],
        ["Épocas",     str(len(historial)),            ""],
    ]
    tbl = axes[2].table(cellText=datos,
                        colLabels=["Métrica", "Valor", "Mejor"],
                        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.7)
    for j in range(3):
        tbl[0, j].set_facecolor("#2E75B6")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Resaltar filas de accuracy y ECE
    for fila in [1, 2]:
        for j in range(3):
            tbl[fila, j].set_facecolor("#FFF3CD")

    plt.tight_layout()
    ruta = f"resultados_grid/figuras/{nombre}.png"
    plt.savefig(ruta, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  → Figura guardada: {ruta}")


# ═══════════════════════════════════════════════════════════════
# FIGURA RESUMEN FINAL
# ═══════════════════════════════════════════════════════════════

def plot_resumen(resultados):
    """
    Heatmaps de accuracy y ECE en función de lr y prior_sigma.
    Permite ver de un vistazo qué combinación es mejor.
    """
    lrs    = sorted(set(r["lr"] for r in resultados.values()))
    sigmas = sorted(set(r["prior_sigma"] for r in resultados.values()))

    def matriz(metrica):
        M = np.zeros((len(sigmas), len(lrs)))
        for i, ps in enumerate(sigmas):
            for j, lr in enumerate(lrs):
                match = [r for r in resultados.values()
                         if r["lr"] == lr and r["prior_sigma"] == ps]
                if match:
                    M[i, j] = match[0][metrica]
        return M

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Grid Search BNN — Heatmaps de métricas clave", fontsize=13)

    lr_labels = [str(lr) for lr in lrs]
    ps_labels = [str(ps) for ps in sigmas]

    metricas_hm = [
        ("accuracy",   "Accuracy ↑",   "Blues",   False),
        ("ece",        "ECE ↓",         "Reds_r",  True),
        ("entropia_id","H_id ↓",        "Oranges_r", True),
    ]

    for ax, (metrica, titulo, cmap, invertir) in zip(axes, metricas_hm):
        # H_id no está directamente — extraerlo
        if metrica == "entropia_id":
            M = np.zeros((len(sigmas), len(lrs)))
            for i, ps in enumerate(sigmas):
                for j, lr in enumerate(lrs):
                    match = [r for r in resultados.values()
                             if r["lr"] == lr and r["prior_sigma"] == ps]
                    if match:
                        M[i, j] = match[0]["H_id"]
        else:
            M = matriz(metrica)

        im = ax.imshow(M, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(lrs)));    ax.set_xticklabels(lr_labels, fontsize=9)
        ax.set_yticks(range(len(sigmas))); ax.set_yticklabels(ps_labels, fontsize=9)
        ax.set_xlabel("Learning rate"); ax.set_ylabel("prior_sigma")
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Anotar valores en cada celda
        for i in range(len(sigmas)):
            for j in range(len(lrs)):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if M[i,j] > M.mean() else "black")

        # Marcar la mejor celda con un borde
        if not invertir:
            best = np.unravel_index(M.argmax(), M.shape)
        else:
            best = np.unravel_index(M.argmin(), M.shape)
        rect = plt.Rectangle((best[1]-0.5, best[0]-0.5), 1, 1,
                              linewidth=3, edgecolor="gold", facecolor="none")
        ax.add_patch(rect)

    plt.tight_layout()
    ruta = "resultados_grid/resumen_convergencia.png"
    plt.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✓ Figura resumen guardada: {ruta}")


# ═══════════════════════════════════════════════════════════════
# GRID SEARCH PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  GRID SEARCH — BNN PyTorch (foco: ECE y convergencia)")
    print(f"  LR: {LEARNING_RATES}")
    print(f"  prior_sigma: {PRIOR_SIGMAS}")
    print(f"  Total: {len(LEARNING_RATES) * len(PRIOR_SIGMAS)} combinaciones")
    print("="*60 + "\n")

    resultados = {}

    for lr, ps in itertools.product(LEARNING_RATES, PRIOR_SIGMAS):
        nombre = f"lr{lr}_ps{ps}".replace(".", "")
        print(f"\n── lr={lr}  prior_sigma={ps} ──────────────────")

        t0 = time.time()
        torch.manual_seed(SEED)
        modelo = ModeloBNN(prior_sigma=ps)
        historial = entrenar(modelo, lr=lr)

        print("  Calculando métricas ...")
        m_id  = calcular_metricas(modelo, test_loader)
        m_ood = calcular_metricas(modelo, ood_loader)
        elapsed = time.time() - t0

        resultado = {
            "lr":          lr,
            "prior_sigma": ps,
            "epocas":      len(historial),
            "tiempo_s":    round(elapsed, 1),
            "accuracy":    round(m_id["accuracy"], 4),
            "ece":         round(m_id["ece"], 4),
            "H_id":        round(m_id["entropia_media"], 4),
            "H_ood":       round(m_ood["entropia_media"], 4),
            "separacion":  round(m_ood["entropia_media"] - m_id["entropia_media"], 4),
            "log_score":   round(m_id["log_score"], 4),
            "brier_score": round(m_id["brier_score"], 4),
        }
        resultados[nombre] = resultado

        print(f"  Acc={resultado['accuracy']:.4f}  ECE={resultado['ece']:.4f}  "
              f"H_id={resultado['H_id']:.4f}  H_ood={resultado['H_ood']:.4f}  "
              f"({len(historial)} épocas  {elapsed:.0f}s)")

        plot_combinacion(historial, m_id, m_ood, lr, ps, nombre)

    # Guardar JSON
    with open("resultados_grid/resultados.json", "w") as f:
        json.dump(resultados, f, indent=2)
    print("\n✓ Resultados guardados en resultados_grid/resultados.json")

    # Figura resumen con heatmaps
    plot_resumen(resultados)

    # ── Resumen en consola ───────────────────────────────────
    print("\n─── MEJORES COMBINACIONES ───────────────────────────")
    mejor_acc = max(resultados.values(), key=lambda r: r["accuracy"])
    mejor_ece = min(resultados.values(), key=lambda r: r["ece"])
    mejor_sep = max(resultados.values(), key=lambda r: r["separacion"])

    print(f"\n  ★ Mejor Accuracy:    lr={mejor_acc['lr']:6}  σ={mejor_acc['prior_sigma']}  "
          f"→ Acc={mejor_acc['accuracy']:.4f}  ECE={mejor_acc['ece']:.4f}")
    print(f"  ★ Mejor ECE:         lr={mejor_ece['lr']:6}  σ={mejor_ece['prior_sigma']}  "
          f"→ ECE={mejor_ece['ece']:.4f}  Acc={mejor_ece['accuracy']:.4f}")
    print(f"  ★ Mejor Sep ID/OOD:  lr={mejor_sep['lr']:6}  σ={mejor_sep['prior_sigma']}  "
          f"→ sep={mejor_sep['separacion']:.4f}  Acc={mejor_sep['accuracy']:.4f}")

    # Advertir si la mejor combinación por ECE no ha convergido bien
    if mejor_ece["epocas"] < 30:
        print(f"\n  ⚠️  La mejor combinación por ECE solo corrió {mejor_ece['epocas']} épocas.")
        print(f"      Considera aumentar EPOCHS para verificar que realmente ha convergido.")

    print("\n" + "="*60)
    print("  GRID SEARCH COMPLETADO")
    print(f"  Figuras en: resultados_grid/figuras/")
    print(f"  Resumen:    resultados_grid/resumen_convergencia.png")
    print("="*60 + "\n")

    # Descargar en Colab automáticamente
    try:
        import shutil
        shutil.make_archive("resultados_grid", "zip", "resultados_grid")
        from google.colab import files
        files.download("resultados_grid.zip")
        print("✓ Descarga iniciada automáticamente")
    except Exception:
        print("  (Para descargar manualmente ejecuta las últimas 3 líneas del script)")


if __name__ == "__main__":
    main()