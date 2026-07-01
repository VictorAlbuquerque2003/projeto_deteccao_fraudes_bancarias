# 🔍 Detecção de Fraude em Cartões de Crédito

## 📌 Sobre o Projeto

Este projeto aplica técnicas de Machine Learning para detecção de fraudes em transações de cartão de crédito utilizando o dataset **Credit Card Fraud Detection**.

⚠️ Este trabalho tem origem em um exercício de um curso de Machine Learning. O objetivo inicial do curso era implementar um modelo básico de detecção de fraudes. A proposta deste projeto foi **evoluir o código original**, aplicando boas práticas de engenharia de software, melhor organização do código e inclusão de novos modelos e técnicas de interpretação.

O foco principal desta evolução foi transformar um script simples em um projeto estruturado, modular e mais próximo de um ambiente profissional.

---

## 🚀 Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Matplotlib
- SHAP

---

## 📂 Estrutura do Projeto

```text
.
├── data
│   └── creditcard.csv
│
├── src
│   ├── app.py
│   ├── models.py
│   └── visualization.py
│
├── original_code
│   └── course_version.py
│
└── README.md
```

## 📊 Dataset

Este projeto utiliza o dataset **Credit Card Fraud Detection**.

O dataset pode ser obtido em:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Após o download, coloque o arquivo `creditcard.csv` dentro da pasta:

data/

## 🔄 Melhorias em relação ao código original

Este projeto foi desenvolvido a partir de um exercício de curso, cujo código original consistia em um único script com um modelo básico de classificação.

Durante a evolução do projeto, foram realizadas as seguintes melhorias:

### ⚖️ Tratamento de desbalanceamento de classes

O dataset apresenta forte desbalanceamento entre classes (fraudes vs transações normais).

Foram testadas duas abordagens:

- Undersampling
- Oversampling (SMOTE)

O **SMOTE apresentou melhor desempenho em recall sem uma perda considerável de precision**, sendo mais adequado para o problema, já que o objetivo principal é reduzir falsos negativos (fraudes não detectadas).

---

### 📊 Feature Engineering e seleção de variáveis

Foi aplicada transformação logarítmica na variável `Amount` para reduzir a assimetria da distribuição.

Durante a análise do pipeline, optou-se por remover as variáveis relacionadas a valor da transação (`Amount` e `Amount_log`), uma vez que:

- representam essencialmente a mesma informação em diferentes escalas
- poderiam introduzir redundância no conjunto de features

Essa decisão simplifica o espaço de entrada e reduz redundância de informação no modelo.

---

### 🧱 Refatoração da arquitetura do projeto

O código original foi reestruturado para melhorar legibilidade e manutenção:

- Separação do código em módulos:
  - `models.py` → treinamento dos modelos
  - `visualization.py` → funções de visualização
  - `app.py` → pipeline principal

Essa organização torna o projeto mais próximo de um fluxo de trabalho real em projetos de Machine Learning.

---

## 📊 Comparação de Modelos

Os modelos foram avaliados com foco principal em **recall** sem a perda significativa de precision, devido à natureza do problema de detecção de fraudes, onde falsos negativos são mais críticos do que falsos positivos.

Abaixo está o desempenho dos modelos testados:

| Modelo               | Precision | Recall | F1-Score | ROC AUC |
|---------------------|----------|--------|---------|--------|
| Regressão Logística | 0.81     | 0.66   | 0.72    | 0.945  |
| Random Forest       | 0.56     | 0.84   | 0.67    | 0.977  |
| XGBoost             | 0.86     | 0.82   | 0.84    | 0.982  |

### 🏆 Melhor Modelo

O **XGBoost** apresentou o melhor desempenho geral, com o maior valor de ROC AUC e o melhor equilíbrio entre precision e recall.

Ele se mostrou mais eficiente na detecção de fraudes, reduzindo falsos negativos sem comprometer significativamente a precisão.