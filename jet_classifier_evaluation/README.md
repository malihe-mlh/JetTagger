# Jet Classifier Evaluation

This directory contains scripts for evaluating jet classifier performance in top quark analysis using deep learning models.

## Scripts Overview

### Performance Metrics
1. **`confusion_matrix.py`** - Confusion matrix showing classifier predictions vs. true labels
2. **`roc_curve.py`** - Receiver Operating Characteristic (ROC) curves for all jet flavor classes
3. **`mistag_rate_SignalEfficiency.py`** - Mistag rate vs. signal efficiency curves for optimization

### Score Distributions
4. **`THAD_score_stacked_normalized.py`** - Stacked histogram of THAD classifier scores (normalized, shows composition)
5. **`TLEP_score_stacked_normalized.py`** - Stacked histogram of TLEP classifier scores (normalized, shows composition)
6. **`score_distribution_THAD.py`** - Overlay step histograms of THAD scores for all flavors (normalized, compares shapes)
7. **`score_distribution_TLEP.py`** - Overlay step histograms of TLEP scores for all flavors (normalized, compares shapes)


# Run individual scripts
python confusion_matrix.py
python roc_curve.py
python mistag_rate_SignalEfficiency.py
python THAD_score_stacked_normalized.py
python score_distribution_THAD.py
python TLEP_score_stacked_normalized.py
python score_distribution_TLEP.py
