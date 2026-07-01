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

## Dataset

Este projeto utiliza o dataset **Credit Card Fraud Detection**.

O dataset pode ser obtido em:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Após o download, coloque o arquivo `creditcard.csv` dentro da pasta:

data/