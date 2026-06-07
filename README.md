# BNN-TFG: Comparativa de modelos bayesianos para cuantificación de incertidumbre
 
Código del Trabajo de Fin de Grado *"Redes neuronales bayesianas frente a modelos deterministas: análisis de la incertidumbre en clasificación de imágenes"* (Grado en Ingeniería Informática, CUNEF Universidad).
 
Autor: Luca Scotti · Director: Roi Naveiro
 
El trabajo compara tres enfoques de modelado con distinto tratamiento de la incertidumbre —un modelo determinista, MC Dropout y una red neuronal bayesiana entrenada con Bayes by Backprop— sobre una tarea de clasificación de imágenes, evaluando su comportamiento tanto dentro de la distribución (MNIST) como fuera de ella (Fashion-MNIST).
 
## Contenido
 
- `experimento_principal.py` — Experimento comparativo entre el modelo determinista, MC Dropout y la BNN sobre MNIST (ID) y Fashion-MNIST (OOD). Entrena los tres modelos, calcula todas las métricas y genera las figuras del trabajo.
- `grid_search_bnn.py` — Búsqueda en cuadrícula sobre el learning rate y la desviación estándar del prior para seleccionar los hiperparámetros de la BNN. Genera los heatmaps y los resultados numéricos de cada métrica.
## Arquitectura
 
Los tres modelos comparten la misma CNN base, de forma que las diferencias se atribuyan al enfoque de modelado y no a la arquitectura:
 
```
Input 28x28x1
Conv2d(1 -> 16, k=5)  -> ReLU -> MaxPool(2)
Conv2d(16 -> 32, k=5) -> ReLU -> MaxPool(2)
Flatten (32*4*4 = 512)
Linear(512 -> 128) -> ReLU
Linear(128 -> 10)  -> Softmax
```
 
La única diferencia entre modelos está en cómo se parametrizan los pesos:
 
- **Determinista**: capas `Conv2d` y `Linear` estándar de PyTorch (pesos puntuales).
- **MC Dropout**: misma arquitectura con una capa `Dropout(p=0.3)` entre la primera capa densa y la salida, activa también en inferencia.
- **BNN**: cada capa convolucional y lineal se sustituye por su equivalente bayesiana, con pesos `q(w) = N(mu, sigma^2)`, truco de la reparametrización y `sigma = log(1 + exp(rho))` (softplus).
## Conjuntos de datos
 
- **MNIST** (ID): 60 000 imágenes de entrenamiento, 10 000 de test. Normalización con media 0.1307 y desviación 0.3081.
- **Fashion-MNIST** (OOD): mismo formato (28x28, escala de grises, 10 clases), nunca visto en entrenamiento. Se usa para evaluar la incertidumbre ante entradas no familiares.
Ambos se descargan automáticamente vía `torchvision` la primera vez que se ejecuta el script.
 
## Configuración exacta de hiperparámetros
 
**Determinista y MC Dropout**
 
| Hiperparámetro | Valor |
|---|---|
| Optimizador | Adam |
| Learning rate | 1e-3 |
| Épocas máximas | 50 |
| Early stopping (paciencia) | 5 épocas |
| Tasa de dropout (solo MC Dropout) | p = 0.3 |
| Pasadas estocásticas en inferencia (MC Dropout) | T = 50 |
| Batch size | 128 |
 
**BNN (Bayes by Backprop)**
 
| Hiperparámetro | Valor |
|---|---|
| Optimizador | Adam |
| Learning rate inicial | 1e-2 |
| Prior | N(0, 0.5^2 · I) |
| KL warm-up | beta de 0 a 1 en las primeras 20 épocas |
| Scheduler | ReduceLROnPlateau (activo desde la época 21) |
| Learning rate final | 2.5e-3 (tras bajadas en las épocas 30 y 45) |
| Épocas de entrenamiento | 50 |
| Pasadas estocásticas en inferencia | T = 50 |
| Batch size | 128 |
 
**Búsqueda en cuadrícula de la BNN** (`grid_search_bnn.py`): se exploraron las 9 combinaciones de
`lr ∈ {1e-4, 1e-2, 0.1}` y `sigma_prior ∈ {0.1, 0.5, 1.0}`, bajo el mismo protocolo que el experimento final
(Adam, hasta 50 épocas, batch 128, KL warm-up de 20 épocas, early stopping con paciencia 10 y mejora mínima 0.01),
evaluando accuracy, ECE, entropía media ID/OOD, separación ΔH, log-score y Brier score.
La combinación seleccionada fue **lr = 1e-2** y **prior N(0, 0.5^2)**, por ofrecer el mejor compromiso
entre accuracy en ID y separación de incertidumbre ID/OOD.
 
## Reproducibilidad
 
- Semilla global: `SEED = 42`, fijada en `random`, `numpy` y `torch`, con `cudnn.deterministic = True`.
- Todos los hiperparámetros están definidos como constantes en la cabecera de cada script (`EPOCHS`, `LR_DET`, `LR_BNN`, `T_MC`, `BATCH_SIZE`).
## Requisitos
 
```bash
pip install torch torchvision matplotlib seaborn scipy numpy
```
 
> El script importa `seaborn` además de las demás; asegúrate de que está en `requirements.txt`.
 
## Uso
 
Recomendado en Google Colab con GPU activada (una T4 gratuita entrena el experimento en pocos minutos a 20 épocas; el entrenamiento completo a 50 épocas con la BNN es más lento):
 
```bash
python experimento_principal.py
```
 
Para reproducir la selección de hiperparámetros de la BNN:
 
```bash
python grid_search_bnn.py
```
 
Las figuras se guardan en `resultados/figuras/` y las métricas numéricas en `resultados/`.
 
## Métricas calculadas
 
Accuracy, entropía predictiva (ID y OOD) y su separación ΔH = H_ood − H_id, ECE (Expected Calibration Error), log-score, selective accuracy y curvas de rechazo automático por umbral de entropía.
 
> El ECE se calcula con **10 bins** de confianza uniformes. Tenlo en cuenta al comparar con valores de la literatura que usen otra cantidad de bins.
 
## Resultados principales
 
| Modelo | Accuracy | ECE | ΔH (ID/OOD) |
|---|---|---|---|
| Determinista | 0.9915 | 0.0057 | 0.457 |
| MC Dropout | 0.9927 | 0.0029 | 0.783 |
| BNN | 0.8633 | 0.1905 | 0.718 |
 
La BNN no alcanza convergencia completa en 50 épocas (de ahí su menor accuracy y su ECE elevado por infraconfianza), pero es el modelo que más separa la entropía entre datos ID y OOD, junto a MC Dropout. Los detalles y el análisis completo están en la memoria.
 
## Referencias principales
 
- C. Blundell et al., *Weight Uncertainty in Neural Networks*, ICML 2015 (Bayes by Backprop).
- Y. Gal y Z. Ghahramani, *Dropout as a Bayesian Approximation*, ICML 2016 (MC Dropout).
- C. Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 (ECE).
