# README — Forecast Semanal PDV/SKU (Hackathon)

Este projeto treina uma LSTM para prever **quantidade semanal por PDV/SKU** nas **5 semanas de janeiro/2023** usando o histórico de **2022**. O pipeline está organizado em células (notebook) e foi escrito para ser **reprodutível**, **rápido** e com **sanidade contra vazamentos**.

---

## 1) Pré-requisitos

- Python 3.9+ (recomendado 3.10/3.11)
- Pip/venv ou Conda
- (Opcional, recomendado) GPU com CUDA para treinar mais rápido

### Instalação

```bash
# recomenda-se usar um virtualenv
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib tensorflow
```

> Caso tenha problemas com `tensorflow` em CPU/Windows, instale a versão específica compatível com sua plataforma.

---

## 2) Dados de entrada

Você precisa consolidar as 3 bases (transações, produtos, PDVs) em um único DataFrame `df_final`, depois criar `df_modelo_inicial` com as colunas mínimas:

- `transaction_date` (datetime)
- `pdv` (str)
- `produto` (str)
- `quantity` (float)

A seguir, agregue **por semana ISO** e crie o dataset final (`df_modelo_inicial`) com schema:

| coluna      | tipo     |
|-------------|----------|
| pdv         | object   |
| produto     | object   |
| ano_iso     | int32    |
| semana      | int32    |
| quantity    | float64  |

**Semanas ISO** (exemplo):
```python
df_modelo_inicial['ano_iso']  = df_modelo_inicial['transaction_date'].dt.isocalendar().year.astype(int)
df_modelo_inicial['semana']   = df_modelo_inicial['transaction_date'].dt.isocalendar().week.astype(int)
```

---

## 3) Estrutura do projeto

- **Notebook** (recomendado): executar as células na ordem.
- **Células**:
  1. **Setup & Config** — imports, seeds, hiperparâmetros.
  2. **Preparação (`df_nn`)** — limpeza, agregação de duplicatas e criação de `week_start`.
  3. **Split temporal** — 80/20 por semanas ISO com `gap=1`.
  4. **Escala por série** — MinMax **fit somente no treino** (anti-vazamento).
  5. **Funções utilitárias** — janelas, (des)escala, WMAPE, limpeza.
  6. **Treino LSTM** — janelas, sample weights ~|y|, early stopping, seleção do melhor.
  7. **Forecast + CSV** — rollout para ISO‐semanas 1..5 de 2023; salva `previsoes_jan2023.csv`.

---

## 4) Execução passo a passo

### 4.1. Setup & Config
Defina:
- `GAP = 1`
- `TRAIN_FRAC = 0.80`
- `RANDOM_SEED = 42`
- `CONFIGS` (ex.: `lookback=8`, `units=64`, `dropout=0.2`, `loss='huber'`, `lr=1e-3`, `batch_size=1024`, `epochs=12`)

### 4.2. Preparação (`df_nn`)
- Colunas: `['pdv','produto','ano_iso','semana','quantity']`
- Somar duplicatas por `(pdv,produto,ano_iso,semana)`
- Criar `week_start` (segunda-feira ISO): `pd.to_datetime(f"{ano}{semana:02d}1", format='%G%V%u')`
- Ordenar por `['pdv','produto','week_start']`

### 4.3. Split temporal (80/20 + gap)
- Construir `weeks_train` e `weeks_test` com `TimeSeriesSplit`
- Marcar `df_nn['split'] ∈ {train, test}`

### 4.4. Escala por série (MinMax do treino)
- Por `(pdv,produto)` **no treino**, calcular `ymin`, `ymax` e `range=(ymax-ymin).replace(0,1.0)`
- Fallback global (min/max do treino) para séries só no teste
- `quantity_scaled = (quantity - ymin)/range`

### 4.5. Utilitários
- `build_windows_intrasplit_with_meta` — janelas de treino/val (contíguas)
- `build_windows_cross_boundary_with_meta` — janelas de teste (alvo em `test`)
- `inverse_per_series_from_meta` — desescala por série
- `wmape` — métrica oficial
- `clean_windows` — remove janelas com NaN/Inf

### 4.6. Treino LSTM
- (Recomendado p/ reprodutibilidade do melhor score):
  - Filtrar séries com **≥ 16 semanas**.
  - Cap de janelas de treino: `MAX_TRAIN_WINDOWS = 800_000` (determinístico).
- Pesos por amostra `~ |y_real|` (aproxima WMAPE).
- Modelo:
  ```
  Input (lookback,1) → LSTM(units, dropout)
  → Dense(units/2, ReLU) → Dense(1, linear)
  ```
- Callbacks: `EarlyStopping(patience=2, restore_best_weights=True)` + `ReduceLROnPlateau`.
- Salvar `best_model`, `best_lookback` e (opcional) `best_history` (curvas).

### 4.7. Forecast + CSV
- Construir janelas iniciais (apenas histórico ≤ fim do treino).
- Rollout **5 passos** (semanas ISO **1..5/2023**).
- Desescalar por série e salvar **`previsoes_jan2023.csv`** (separador `;`, UTF-8, colunas: `semana;pdv;produto;quantidade`).

---

## 5) Avaliação

- **WMAPE global** (teste) e **baseline lag-1**; cobertura (% de casos em que o modelo erra menos que o baseline).
- **Backtesting** (splits rolantes em 2022): mediana/IQR do WMAPE.
- **Erro por segmento** (faixa de volume, PDV, produto, semana do ano).

---

## 6) Reprodutibilidade

- Seeds fixas: `np.random.seed(42)` e `tf.keras.utils.set_random_seed(42)`
- Mantenha iguais: filtros (≥16 semanas), cap de janelas (800k) e `CONFIGS`.

---

## 7) Dicas de performance

- Use GPU (CUDA) quando possível.
- Ajuste `MAX_TRAIN_WINDOWS` e `batch_size` conforme memória.
- `clean_windows` evita NaN/Inf que derrubam o treino.

---


## 8) Saída esperada

- Logs com formas de janelas e métricas.
- **`previsoes_jan2023.csv`** com ~N_series × 5 linhas no formato exigido.

---
