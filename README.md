Dataset Source: https://drive.google.com/drive/folders/1Aihn49cUYMjCgeCTFBTyprjrgZO3UY6r?usp=drive_link

# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Project Objective

This project builds and analyses a feed-forward neural network for a supervised customer churn prediction problem. The focus is not only on model performance, but also on showing how a neural network learns through the forward pass, loss calculation, backpropagation, and parameter updates.

## Dataset

- File used: `data/customer_churn_nn.csv`
- Target column: `churn`
  - `1` = customer churned
  - `0` = customer retained
- Identifier removed before modelling: `customer_id`
- Categorical features encoded: `region`, `plan_type`, `contract_type`, `payment_method`
- Numerical features scaled before training.

## Repository Structure

```text
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb
├── requirements.txt
├── data/
│   ├── customer_churn_nn.csv
│   └── data_dictionary.md
└── results/
    ├── model_comparison_table.csv
    ├── model_comparison_table.png
    └── evaluation_outputs.png
```

## Method Summary

1. Loaded and explored the dataset.
2. Checked rows, columns, data types, missing values, statistical summary, and target distribution.
3. Removed `customer_id` because it is an identifier, not a meaningful predictive feature.
4. Encoded categorical variables using one-hot encoding.
5. Scaled numerical variables using standardisation.
6. Used a stratified train/test split so the rare churn class is represented in both sets.
7. Built a PyTorch feed-forward neural network with:
   - Input layer matching the number of processed features
   - One or more hidden layers
   - ReLU or Tanh activation
   - Single output logit for binary churn prediction
   - Binary cross-entropy with logits loss
   - Adam optimizer
8. Ran five hyperparameter experiments and compared training/testing metrics.

## Key Result

The dataset is highly imbalanced, with far more retained customers than churned customers. Because of that, accuracy alone is misleading. The notebook reports accuracy, precision, recall, F1 score, ROC-AUC, loss, and the confusion matrix.

The final model should not be interpreted as production-ready. It is mainly an academic demonstration of neural network training behaviour and the effect of hyperparameter changes.

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Open and run:

```bash
jupyter notebook notebook.ipynb
```

The notebook saves output files in the `results/` folder.
