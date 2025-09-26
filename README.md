# 🤖 Perceptron – Aspirador Inteligente (AI 2025 • UNIBH)

**Disciplina:** Inteligência Artificial – Trilha Computação  
**Professores:** Fabrício Valadares, Alexandre "Montanha"

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
- **Tipo de piso**: codificado como inteiro (`1`, `2`, `3`).
- **Sujeira**: escala original **`0–10`** (mesmo que o enunciado cite `1–5`, a base fornecida e o código aceitam de `0–10`).
- **Distância do obstáculo**: `0–5` metros.

Todas as entradas são **normalizadas** para a faixa `[0,1]` no pré-processamento:
- Piso: `piso / 3.0`
- Sujeira: `poeira / 10.0`
- Distância: `obstaculos / 5.0`

### Saídas
- **Velocidade** (`1–5`)
- **Potência** (`1–3`)

As saídas são tratadas como **variáveis contínuas** (regressão) e reescaladas para seus intervalos físicos após a saída da sigmoid.

---

## 🏗️ Modelo e escolha da ativação

### Arquitetura
- **Perceptron linear multi‑saída** (duas saídas) com bias explícito.
- **Função de ativação:** sigmoid para ambas as saídas.
- As saídas do perceptron são reescaladas para os intervalos de velocidade e potência via:
  ```
  saida = y_min + (y_max − y_min) * sigmoid(z)
  ```

### Por que **sigmoid**?
- Permite gradiente para regressão contínua.
- Facilita o mapeamento direto para os limites físicos das saídas.

---

## 🧪 Dados e treinamento

### Base
- Pequeno conjunto de exemplos manuais, com valores de poeira de `0` a `10`, refletindo a base fornecida pelo professor.

### Pipeline
- **Normalização** das entradas para `[0,1]`.
- **Normalização das saídas** para `[0,1]` antes do treino.
- **Loss:** MSE
- **Otimização:** Gradiente descendente

### Exemplo de execução

```
Resultados:
Piso=2 Poeira=2 Obst=0 -> Velocidade=3.72, Potência=1.02
Piso=1 Poeira=8 Obst=2 -> Velocidade=1.22, Potência=2.99
...
```

O script também mostra a curva de erro (MSE) durante o treinamento.

---

## 📦 Como executar

### Pré‑requisitos
- Python **3.10+**
- `numpy`
- `matplotlib` (para gráfico)

```bash
# (opcional) criar venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install numpy matplotlib
```

### Rodar treinamento + demonstração
```bash
# Executar localmente:
# Abra o notebook Perceptron-AspiradorInteligente.ipynb no Jupyter ou Colab e execute as células
```
Saídas esperadas: logs, métricas, demonstração com os exemplos e gráfico do erro.

---

## 📈 Gráfico de aprendizagem

O notebook plota automaticamente o erro (MSE) por época ao final do treinamento.

---

## 🗂️ Estrutura sugerida do repositório

```
.
├─ Perceptron-AspiradorInteligente.ipynb   # notebook principal
├─ README.md
└─ requirements.txt                        # dependências
```

**Exemplo de `requirements.txt`:**
```
numpy>=1.24
matplotlib>=3.8
```

---

## 🧠 Decisões de projeto (resumo)

- **Regressão contínua** para controle mais realista.
- **Sigmoide + reescala** para respeitar limites físicos.
- **Normalização robusta** para entradas.
- **Visualização do erro** para análise do aprendizado.

---

## 🔬 Limitações e próximos passos

- **Perceptron linear**: não capta relações complexas, podendo ser ampliado para MLP.
- **Base pequena**: expandir exemplos ou coletar dados reais pode melhorar o modelo.
- Explorar **regularização** e **tuning** de hiperparâmetros.

---

## 👥 Time e créditos
Projeto desenvolvido conforme a atividade prática de **IA 2025 – UNIBH**.  
**Autores:** Maria Clara Palhares (Mariacpdb), Kaíky (kaiky-ferreira), Yris (YrisSother), Gabriel (ShugZin), Breno Yohan (Gu4xin), Laysa (Laysa-eSerrão).
