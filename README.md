# Predicting Coral Reef Regimes from Human and Natural Influences

## 1. Problem Statement

Predict the ecological regime (1, 2, 3 or 5) of a coral reef from 20 human and natural predictors (e.g. effluent, sedimentation, wave energy, chlorophyll, PAR, reef complexity, depth). This is a four-class classification problem based on the work of Cheng, Murphy and Walecka (2019).

## 2. Dataset

- **Source:** [Hawaii_RegimesPredictors](https://github.com/salliewalecka/Hawaii_RegimesPredictors)
- **Size:** 620 reef sites, 20 predictors, target `Regime` ∈ {1, 2, 3, 5}
- **Missing values:** `Complexity` (113), `Depth` (12)
- **Split:** 80% train (496) / 20% test (124), random seed 47

The dataset is downloaded automatically by the notebook; no manual download is required (internet access needed).

## 3. Approach

1. **Missing-value handling:** dropping rows, mean imputation, Lasso regression, and a custom K-means cluster imputer.
2. **Feature engineering:** five additional features derived from wave energy, complexity, depth, invasive algae, PAR and anomaly variables.
3. **Models:** Majority-class baseline, Logistic Regression, SVM (RBF, polynomial), Random Forest, Gradient Boosting.
4. **Evaluation:** `GridSearchCV` with 5-fold stratified cross-validation on the training set (micro-F1, macro-F1); the test set is used once for the final model.
5. **Analysis:** confidence calibration, confusion matrices, PCA visualisation, feature importances, comparison with the published paper.

## 4. Results

| Model | CV micro-F1 |
|---|---|
| **Random Forest (original features) – selected** | **0.688** |
| Random Forest (engineered features) | 0.684 |
| SVM RBF (engineered features) | 0.678 |
| Gradient Boosting (engineered features) | 0.673 |
| Logistic Regression (best variant) | ≈ 0.62 |
| Majority-class baseline | 0.272 |

**Final test performance (Random Forest):** micro-F1 = 0.734, macro-F1 = 0.719.

## 5. Repository Structure

```
.
├── Coral_Reef_Reproduction.ipynb   # Complete implementation
├── requirements.txt                # Python dependencies
├── README.md
├── model_comparison.csv            # Generated output
├── paper_comparison.csv            # Generated output
└── test_predictions.csv            # Generated output
```

## 6. Setup and Execution

**Requirements:** Python 3.10+ and an internet connection.

```bash
git clone https://github.com/Chengapa13/ML-Project
cd <repository>

python -m venv venv
source venv/bin/activate        

pip install -r requirements.txt
jupyter notebook Coral_Reef_Reproduction.ipynb
```

In Jupyter, select **Kernel → Restart & Run All**. The three CSV output files are written to the working directory on completion.

**Alternative (Google Colab):** upload `Coral_Reef_Reproduction.ipynb` to [colab.research.google.com](https://colab.research.google.com) and select **Runtime → Run all**.

## 7. Limitations

- Small dataset with spatial dependence between nearby sites.
- Regime labels are derived from a prior clustering study.
- Limited hyperparameter search.
- Feature importance indicates association, not causation.

## 8. References

1. Cheng, T., Murphy, A., Walecka, S. (2019). *Predicting Coral Reef Regimes from Human and Natural Influences.* [Stanford CS229](https://cs229.stanford.edu/proj2019spr/report/56.pdf).
2. [Authors' source code and data](https://github.com/salliewalecka/Hawaii_RegimesPredictors)
