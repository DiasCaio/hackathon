# Previsão de Vendas Semanais por PDV × SKU (Jan/2023)

Projeto para prever **quantidade semanal** por **PDV × SKU** para as 5 semanas de **janeiro/2023**, usando o histórico de **2022**. O objetivo é apoiar reposição no varejo, com foco em **WMAPE** como métrica principal.

---

## Sumário
- [Stack / Ambiente](#stack--ambiente)
- [Escolha de GPU](#escolha-de-gpu)
- [Dados](#dados)
- [Pipeline (visão geral)](#pipeline-visão-geral)
- [Como rodar no Kaggle](#como-rodar-no-kaggle)
- [Saídas (formato de submissão)](#saídas-formato-de-submissão)
- [Decisões de Modelagem (racional)](#decisões-de-modelagem-racional)
- [Validação & Métrica](#validação--métrica)
- [Compactação p/ Submissão](#compactação-p-submissão)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Solução de problemas](#solução-de-problemas)
- [Licença](#licença)

---

## Stack / Ambiente

- Python 3.11 (Kaggle)
- **Pandas**, **NumPy**, **Matplotlib**
- **XGBoost 2.x** (GPU)
- **Numba** (para aceleração de cálculos)
- Parquet via **pyarrow**

> No Kaggle isso já vem pré-instalado.

---

## Escolha de GPU

Usamos **GPU T4 (2× disponível no Kaggle)**. O treino com XGBoost utiliza **uma GPU** por padrão (multi-GPU exigiria Dask/Rabit e não traz ganho proporcional aqui). A escolha T4 é estável e tem boa memória; **P100** também funciona, mas mantivemos T4 pela disponibilidade.

---

## Dados

Fonte: dataset Kaggle anexado como **`arquivos`** (três Parquets).

- **Transações (2022)**  
  `internal_store_id, internal_product_id, transaction_date, quantity, net_value, ...`
- **Cadastro de PDVs**  
  `pdv, premise (On/Off), categoria_pdv, zipcode`
- **Cadastro de Produtos**  
  `produto, categoria, descricao, tipos, marca, fabricante, ...`

> **Importante:** os nomes dos arquivos possuem sufixos dinâmicos (ex.: `tid-...`). O notebook **detecta automaticamente** qual é cada arquivo pelo **esquema de colunas** (sem hardcode de nomes).

---

## Pipeline (visão geral)

1. **Ingestão & Normalização**
   - Leitura Parquet (auto-descoberta por colunas).
   - `lower_snake_case` nos nomes de colunas.

2. **Agregação Semanal**
   - Granularidade alvo: **PDV × SKU × semana**.
   - Soma de `quantidade` e `net_value` por semana.

3. **EDA (sanity checks)**
   - Histograma de `quantidade` (cap 99º pct para visualização).
   - Média por `weekofyear`.
   - Proporção de zeros por semana.

4. **Limpeza + Features (vetorizado)**
   - **Negativos → 0** e **cap global p99.9** em `quantidade`.
   - **Preço unitário** robusto: `uprice = net_value / quantidade` (com proteção a zero).
   - **Calendário cíclico**: `sin/cos(weekofyear)`.
   - **Lags**: `lag_1,2,3,4,8,12`, **rollings**: `mean_4,8,12`.
   - **WSL** (weeks_since_last_sale) com **Numba** (1 passada O(n)).
   - **Agregados globais com lag**: `sku_week_avg_lag1`, `pdv_week_total_lag1`.
   - IDs densas (`pdv_id`, `prod_id`) usando `category.codes`.

5. **Split Temporal**
   - **Treino:** até **2022-12-04**  
   - **Validação (holdout):** **dez/2022**  
   - Evita vazamento e simula o cenário de prever jan/2023.

6. **Treino do Modelo (XGBoost GPU)**
   - `objective="reg:absoluteerror"` (MAE), **alinha melhor com WMAPE**.
   - `tree_method="gpu_hist"`, `DeviceQuantileDMatrix(max_bin=128)`.
   - Regularização (`max_depth=7`, `min_child_weight≈8`, `lambda`, `alpha`).
   - `subsample`/`colsample_bytree` para reduzir variância.
   - **Early stopping** com WMAPE em validação.

7. **Forecast de 5 semanas (Jan/2023)**
   - Previsão **semana a semana** com **feedback** (atualiza lags/WSL/agregados com as próprias previsões).
   - Clamp em p99.9 e arredondamento para inteiros.

8. **Compactação p/ Submissão**
   - Mantém pares com **venda nas últimas 12 semanas**.
   - Seleciona **top-K** por volume recente para ficar dentro dos limites do portal.

---

## Como rodar no Kaggle

1. Crie um **Notebook** e adicione o dataset **`arquivos`** (Add data → “arquivos”).  
2. Em **Settings**, selecione **GPU → T4 x2** (ok rodar em 1 T4 também).  
3. Copie o notebook deste repo (ou suba o `.ipynb`).  
4. **Execute** as células em ordem.  
   - O notebook faz a **detecção automática** dos Parquets.  
   - O output final é salvo em:
     - `/kaggle/working/forecast_jan2023_refined_small.csv`  
     - `/kaggle/working/forecast_jan2023_refined_small.parquet`

---

## Saídas (formato de submissão)

CSV/Parquet com **UTF-8** e **`";"`** como separador (CSV), colunas:

```
semana;pdv;produto;quantidade
1;1023;123;120
2;1045;234;85
...
```

**Restrições do portal** (já respeitadas na compactação):
- CSV **< 50 MB**
- Parquet **≤ 1.5M linhas**

---

## Decisões de Modelagem (racional)

- **Agregação semanal:** reduz ruído diário e casa com o **ciclo de reposição** do varejo.
- **Cap p99.9 + negativos→0:** protege o modelo de **outliers**/erros sem achatar a distribuição.
- **Sinais de calendário cíclicos:** `sin/cos(weekofyear)` evitam a “quebra” entre semanas 52→1.
- **Lags & Rollings:** capturam **memória recente** e tendências (1–12 semanas).
- **WSL (Numba):** diferencia **zero pontual** de “**faz semanas sem vender**” (demanda intermitente).
- **Agregados globais (lag1):** contexto de SKU/PDV na semana **sem vazamento**.
- **XGBoost (MAE):** mais próximo do **WMAPE**, robusto a cauda longa e fácil de acelerar na GPU.

---

## Validação & Métrica

- **WMAPE** (Weighted Mean Absolute Percentage Error):
  
  \[ \mathrm{WMAPE} = rac{\sum |y - \hat{y}|}{\sum |y|} \]

- Holdout: **dez/2022**.  
- O notebook também loga **MAE/WMAPE** em treino/validação e usa **early stopping**.

> Observação: os números exatos variam conforme randomização/ambiente, mas a validação segue o mesmo protocolo.

---

## Compactação p/ Submissão

Para caber nos limites e focar no que gira:
1. Filtra pares com **`last12 > 0`** (houve venda nas 12 últimas semanas observadas).  
2. Ranqueia por `qty_last12` e `qty_total_2022`.  
3. Seleciona **top-K** de pares para manter ~**1.13M linhas** no CSV final (<50MB).

---

## Estrutura do repositório

```
.
├─ notebooks/
│  ├─ notebook-versao-final-hackathon.ipynb              # notebook principal (pipeline completo)
├─ README.md                               # este arquivo

```

---