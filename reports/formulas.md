# Mathematische Formeln aus dem 3D Diffusions-Paper

## Grundnotation

- $B$ — Batchgröße
- $N$ — Anzahl Sensorknoten (z.B. 64, 256, ...)
- $d$ — Dimension pro Knoten (z.B. 5: $[p,\mu,T,E,\sigma]$)
- $T$ — Anzahl Diskrete Zeitschritte (z.B. 1000)
- $N_d = N \times d$ — gestreckte Dimension

## Tensor-Shapes

- **S0**: Shape $(B, N, d)$ — saubere Daten im Batch
- **S_t**: Shape $(B, N, d)$ — verrauschter Zustand bei Zeit $t$
- **A**: Shape $(N, N)$ oder $(N_d, N_d)$ abhängig von Repräsentation
- **L**: Graph Laplacian, Shape $(N,N)$
- **Phi_t**: Action-on-nodes Matrix $(N,N)$
- **Sigma_t**: Kovarianz über gesamte gestreckte Dimension
- **epsilon**: Shape $(B, N, d)$ — standard Normal noise

## Matrizenstruktur (Kronecker-Form)

$$A = \gamma(L \otimes I_d) + \lambda I_{N_d} + B_{block}$$

### Apply-A Operation
```
apply_A(S: (B,N,d)) -> (B,N,d):
gamma * (L @ S_node_features) + lambda * S
```

## Diskrete Zeitevolution (Forward SDE)

### Forward-SDE (kontinuierlich)
$$dS_t = -AS_t dt + \sqrt{2\beta(t)} dW_t$$

mit $A = \gamma(L \otimes I_d) + \lambda I_{N_d}$

### Diskretisierung (explicit Euler)
$$S_{k+1} = (I - A\Delta t)S_k + \sqrt{2\beta(t_k)\Delta t} \xi_k$$

### Kronecker-Struktur Nutzung
$$\Phi(t) = e^{-At} \approx \Phi_{node}(t) \otimes I_d$$

$$\Sigma(t) = \Sigma_{node}(t) \otimes I_d$$

## Präkomputation: Φ(t) und Σ(t)

### Diskretes Φ-Produkt (Euler-Approx)
```
Phi_0 = I_N
for k in 0..T-1:
    Phi_{k+1} = (I_N - A_node * Δt_k) @ Phi_k
```

### Diskrete Σ-Rekursion
```
Sigma_0 = 0
for k in 0..T-1:
    Sigma_{k+1} = (I - A_full*Δt_k) Sigma_k (I - A_full*Δt_k)^T + 2β(t_k)Δt_k * I_full
```

## Tensor-Formeln (Vektorisiert)

### Apply Phi (batch)
```python
# S0: (B,N,d), Phi_node: (N,N)
S_mean = torch.einsum('ij,bjd->bid', Phi_node, S0)
```

### Add Noise (diag Σ approximation)
```python
S_t = S_mean + sigma_t * torch.randn(B,N,d)
epsilon = (S_t - S_mean) / sigma_t
```

### Hat S0 Rekonstruktion
$$\hat{S}_0 = \text{solve\_Phi}(S_t - \sigma_t \epsilon_{hat})$$

## Score-/Loss-Training

### Forward SDE mit Gaussian Lösung
$$S_t | S_0 \sim \mathcal{N}(\Phi(t)S_0, \Sigma(t))$$

$$S_t = \Phi(t)S_0 + \sqrt{\Sigma(t)} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

### Bedingte Log-Dichte
$$\log q(S_t|S_0) = -\frac{1}{2}(S_t - \Phi S_0)^T \Sigma^{-1}(S_t - \Phi S_0) + C$$

### Bedingter Score (Gradient nach S_t)
$$\nabla_{S_t} \log q(S_t|S_0) = -\Sigma^{-1}(S_t - \Phi S_0)$$

$$\nabla_{S_t} \log q = -\Sigma^{-1/2} \epsilon$$

## Loss-Formeln

### ε-MSE (Option A)
$$L_\epsilon = \mathbb{E}_{t,S_0,\epsilon}[\|\epsilon - \epsilon_\theta(S_t,t)\|_2^2]$$

### Direkter Score-Loss (Option B)
$$L_{score} = \mathbb{E}_{t,S_0,q(S_t|S_0)}[\lambda(t) \|s_\theta(S_t,t) - \nabla_{S_t} \log q(S_t|S_0)\|_2^2]$$

$$L_{score} = \mathbb{E}[\lambda(t) \|s_\theta(S_t,t) + \Sigma^{-1/2}\epsilon\|^2]$$

### Score-ε Beziehung
$$s_\theta(S_t,t) \approx -\Sigma^{-1/2} \epsilon_\theta(S_t,t)$$

## Regularisierer (auf $\hat{S}_0$ anwenden)

### Rekonstruktion von $\hat{S}_0$
$$\hat{S}_0(S_t,t) = \Phi(t)^{-1}(S_t - \sqrt{\Sigma(t)} \epsilon_\theta(S_t,t))$$

### Energetische Regularisierung
$$L_{energy} = \mathbb{E}[E(\hat{S}_0)]$$

Quadratisches Potenzial:
$$E(S) = \frac{1}{2}\|M^{1/2}(S - S^*)\|_F^2$$

Ableitung:
$$\nabla_{\hat{S}_0} E = M(\hat{S}_0 - S^*)$$

### Graph-Struktur-Regularisierung
$$L_{graph} = \mathbb{E}[\|L \hat{S}_0\|_F^2]$$

Graph-Laplacian:
$$L = D - A, \quad D_{ii} = \sum_j A_{ij}$$

Kanten-Formulierung:
$$\|L \hat{S}_0\|_F^2 = \sum_{(i,j) \in E} w_{ij} \|\hat{s}_{0,i} - \hat{s}_{0,j}\|^2$$

Ableitung:
$$\nabla_{\hat{S}_0} L_{graph} = 2L^T L \hat{S}_0$$

### Alignierungs-/Kohärenzregularisierung
$$L_{align} = \mathbb{E}\left[\sum_i (1 - \cos(\theta_i))\right]$$

mit
$$\cos(\theta_i) = \frac{\langle v_i, \hat{v}_i \rangle}{\|v_i\| \|\hat{v}_i\|}$$

Ableitung:
$$\nabla_{\hat{S}_0} L_{align} = -\sum_i \frac{\partial \cos(\theta_i)}{\partial \hat{v}_i} \frac{\partial \hat{v}_i}{\partial \hat{S}_0}$$

### Gesamt-Regularisierung
$$L_{reg} = \lambda_E L_{energy} + \lambda_G L_{graph} + \lambda_A L_{align}$$

$$L_{total} = L_{diff} + L_{reg}$$

## Zeitgewichtung λ(t)

- $\lambda(t) = 1$ (Standard)
- $\lambda(t) = 1/\sigma(t)^2$ (Varianz-normiert)
- $\lambda(t) \propto 1/\text{tr}(\Sigma(t))$ (Spur-normiert)

## Reverse Sampling

### Diskrete Übergangsmatrizen
$$F_k = \Phi_{k+1} \Phi_k^{-1}$$
$$Q_k = \Sigma_{k+1} - F_k \Sigma_k F_k^T$$

### Posteriormittelwert und -kovarianz
$$m_{post} = \mu_k + \Sigma_k F_k^T (F_k \Sigma_k F_k^T + Q_k)^{-1} (S_{k+1} - F_k \mu_k)$$
$$\text{cov}_{post} = \Sigma_k - \Sigma_k F_k^T (F_k \Sigma_k F_k^T + Q_k)^{-1} F_k \Sigma_k$$

### Reverse-SDE (für Score-basiert)
$$dS_t = (-AS_t - g(t)^2 s_\theta(S_t,t))dt + g(t)d\bar{W}_t$$

mit $g(t) = \sqrt{2\beta(t)}$

## Numerische Stabilität

### Implicit Euler (stabiler)
$$S_{k+1} = (I + A\Delta t)^{-1} S_k + \sqrt{2\beta\Delta t} (I + A\Delta t)^{-1} \xi$$

### Regularisierung für Invertierung
- Sigma Invertierung: $\Sigma + \epsilon I$ mit $\epsilon = 10^{-6}$
- Phi Invertierung: $\Phi + \epsilon I$ mit $\epsilon = 10^{-9}$

### Low-rank Approximation für Σ
$$\Sigma_{node} \approx U_k U_k^T$$

Sampling:
```python
z = torch.randn((B, r, d))
noise = torch.einsum('ir,brd->bid', U_k, z)
```

## Parameter-Standardwerte

- $B = 16$ (Batchgröße)
- $N = 64$ (Knoten)
- $d = 5$ (Features pro Knoten)
- $T = 1000$ (Zeitschritte, oder 200 für Experimente)
- $\gamma = 0.1$ (Graph-Gewicht)
- $\lambda = 10^{-2}$ (Regularisierung)
- $\beta(t)$: linear schedule von $10^{-5}$ bis $10^{-2}$ über $T$
- Optimizer lr: $10^{-4}$ (Adam)
- $\lambda_E, \lambda_G, \lambda_A = 0.1, 0.1, 0.01$ (Regularisierungs-Gewichte)