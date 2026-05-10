# BNN-TFG: Comparativa de modelos bayesianos para cuantificación de incertidumbre

Código del Trabajo de Fin de Grado "Redes Neuronales Bayesianas e Incertidumbre" 
(Grado en Ingeniería Informática).

## Contenido

- `experimento_principal.py` — Experimento comparativo entre modelo determinista, 
  MC Dropout y BNN sobre MNIST y Fashion-MNIST. Genera todas las figuras y métricas 
  del trabajo.
- `grid_search_bnn.py` — Búsqueda en cuadrícula sobre learning rate y prior_sigma 
  para seleccionar los hiperparámetros óptimos de la BNN.

## Requisitos

```bash
pip install torch torchvision matplotlib scipy numpy
```

## Uso

Ejecutar en Google Colab con GPU activada:

```bash
python experimento_principal.py
```

Los resultados se guardan en `resultados/figuras/`.

## Resultados principales

| Modelo | Accuracy | ECE | ΔH (ID/OOD) |
|---|---|---|---|
| Determinista | 0.9915 | 0.0057 | 0.457 |
| MC Dropout | 0.9927 | 0.0029 | 0.783 |
| BNN | 0.8633 | 0.1905 | 0.718 |