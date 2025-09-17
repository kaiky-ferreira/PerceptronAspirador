# 🤖 Perceptron – Aspirador Inteligente (AI 2025 • UNIBH)

**Disciplina:** Inteligência Artificial – Trilha Computação  
**Prof.:** Fabrício Valadares 

**Local:** Belo Horizonte/MG – Centro Universitário de Belo Horizonte UNiBH

---

## 🎯 Objetivo da atividade

Projetar e treinar **um perceptron para controlar um aspirador de pó inteligente**.  
O modelo recebe três entradas (tipo de piso, nível de sujeira e distância até obstáculos) e retorna **duas saídas contínuas**:
- **Velocidade do movimento** (faixa: `1` a `5`)
- **Potência de sucção** (faixa: `1` a `3`)

A proposta segue o enunciado da atividade prática e inclui **código completo, justificativas de projeto, procedimento de execução, exemplos e validações**.

---

## 🧩 Formulação do problema

### Entradas
- **Tipo de piso**: `madeira`, `cerâmica`, `carpete` (com *aliases* robustos: `wood`, `laminado`, `porcelanato`, `tile`, `carpet`, etc.)  
  Codificação: **one-hot** com 3 posições.
- **Sujeira**: escala original `0–10` (o enunciado cita `1–5`, mas a própria base sugerida usa até `9`; o código aceita `0–10`).  
  Normalização para `[0,1]` por *clamp* e divisão.
- **Distância do obstáculo**: `0–5` metros.  
  Normalização para `[0,1]` por *clamp* e divisão.

> Vetor final de atributos: **5 dimensões** → `[piso_madeira, piso_cerâmica, piso_carpete, sujeira_norm, distancia_norm]`.

### Saídas
- **Velocidade** (`1–5`)
- **Potência** (`1–3`)

As saídas são tratadas como **variáveis contínuas** (regressão) para refletir melhor atuadores reais.

---

## 🏗️ Modelo e escolha da ativação

### Arquitetura
Um **perceptron linear multi‑saída** (duas saídas) com **bias** explícito e ativação de saída **sigmoide limitada** (classe `BoundedSigmoid`).  
Pesos: matriz `W` com forma `(features+1, 2)`.

### Por que **sigmoide** (limitada) e não **step**?
- **Step** gera apenas 0/1 (classificação), inadequado para **faixas contínuas** e controle fino.
- **Sigmoide** permite **gradiente não nulo**, possibilitando **descida de gradiente** com **MSE** e aprendizagem estável.
- A versão **limitada** mapeia diretamente para **intervalos físicos**:  
  `y = y_min + (y_max − y_min) * sigmoid(z)`  
  garantindo **velocidade ∈ [1,5]** e **potência ∈ [1,3]` sem pós-processamento.

**Conclusão:** a **sigmoide limitada** é **a melhor escolha** para este problema de **regressão com faixas**.

---

## 🧪 Dados e treinamento

### Geração de dados (heurística‑alvo)
Para treinar com mais variedade, geramos um dataset sintético a partir de **regras físicas simples**:
- Carpetes pedem **maior sucção** e **menor velocidade**.
- Quanto **maior a sujeira**, **maior a sucção** e **menor a velocidade**.
- Quanto **maior a distância de obstáculos**, **maior pode ser a velocidade**.

> Função: `target_speed_and_suction(...)` produz rótulos contínuos coerentes nessas faixas.

### Pipeline
- **Normalização** de entradas em `[0,1]` (sujeira, distância); **one‑hot** para piso.
- **Loss:** MSE
- **Otimização:** gradiente (taxa `lr=0.2`)
- **Early stopping** com validação hold‑out (`20%`).

### Métricas (execução de exemplo)
```
[FINAL] train MSE=0.0070, MAE=0.0586 | val MSE=0.0070, MAE=0.0598
```
> Erro médio absoluto ~**0,06** nas escalas das saídas → **controle preciso**.

---

## ✅ Sanity checks (validações automáticas)

A função `sanity_checks(model)` verifica:
1. **Faixas** de saída respeitadas: velocidade ∈ `[1,5]`, potência ∈ `[1,3]`.
2. **Monotonicidade aproximada** da potência com sujeira crescente (distância fixa).
3. **Velocidade aumenta com distância** (mantendo sujeira alta).

Esses testes dão segurança de que o modelo aprendeu relações coerentes com o domínio.

---

## 📦 Como executar

### Pré‑requisitos
- Python **3.10+**
- `numpy`

```bash
# (opcional) criar venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install numpy
```

### Rodar treinamento + demonstração
```bash
python main.py
```
Saídas esperadas: logs de época, métricas finais, **demonstração** com cenários de teste (incluindo *aliases* de piso) e execução de **sanity checks**.  
Os **pesos** são salvos em `vacuum_multi.npz`.

### Usar o modelo treinado em outro script
```python
from main import load_model, predict_one
model = load_model("vacuum_multi.npz")
vel, pot = predict_one(model, "carpete", 8, 1.0)
print(vel, pot)
```

---

## 📈 Bônus – Gráfico de aprendizagem (training curve)

Para gerar um gráfico de **MSE por época**, adicione um *callback* simples no laço de treino e plote com `matplotlib`:

```python
# dentro de MultiOutputPerceptron.fit(...)
history = []
...
for ep in range(epochs):
    ...
    tr_loss = self._mse(y_pred, y_true)
    history.append(tr_loss)
    ...
# depois de treinar:
import matplotlib.pyplot as plt
plt.plot(history)
plt.xlabel("Época")
plt.ylabel("MSE (treino)")
plt.title("Curva de aprendizagem")
plt.show()
```

> Dica: também registre `val_loss` para comparar **overfitting** vs **generalização**.

---

## 🗂️ Estrutura sugerida do repositório

```
.
├─ main.py                 # (o arquivo com o código fornecido)
├─ vacuum_multi.npz        # pesos salvos após o treino (gerado)
├─ README.md               # este arquivo
└─ requirements.txt        # opcional: numpy>=1.24
```

**Exemplo de `requirements.txt`:**
```
numpy>=1.24
matplotlib>=3.8  # (se for gerar gráfico bônus)
```

---

## 🧠 Decisões de projeto (resumo)

- **Regressão contínua** (controle mais fino) em vez de classificação discreta.
- **Sigmoide limitada por saída** para impor **restrições físicas** nativamente.
- **Aliases** e **normalização robusta** (com *clamp*) para entradas seguras.
- **Early stopping** orientado por validação para evitar overfitting.
- **Sanity checks** que testam propriedades desejáveis do sistema.

---

## 🔬 Limitações e próximos passos

- **Perceptron linear** capta relações de primeira ordem; interações complexas poderiam se beneficiar de **camadas ocultas** (MLP).
- **Dados sintéticos** baseados em heurística; coletar **telemetria real** do aspirador pode refinar o mapeamento.
- Explorar **regularização** e **otimizadores** (Adam) e **tuning** de hiperparâmetros.
- Implementar **calibração**/quantização se o alvo for um **microcontrolador**.

---

## 👥 Time e créditos
Projeto desenvolvido conforme a atividade prática de **IA 2025 – UNIBH**.  
**Autores:** Maria Clara Palhares (Mariacpdb), Kaíky (kaiky-ferreira), Yris (YrisSother),Gabriel (ShugZin), Breno Yohan (Gu4xin), Laysa (Laysa-eSerrão).

