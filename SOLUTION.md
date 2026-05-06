# SMILES-2026 Hallucination Detection Solution

## Repository

Final repository:  
https://github.com/SofiOnisenko/SMILES-2026-Hallucination-Detection

---

# Introduction

This project focuses on hallucination detection in small language models using hidden-state representations extracted from transformer layers.

The main goal was not only to improve the final AUROC metric, but also to build a solution that generalizes more reliably across different data splits. During development, special attention was paid to reducing overfitting, evaluating stability across folds, and comparing different aggregation strategies for hidden representations.

A large number of experiments showed that increasing model complexity alone did not necessarily improve generalization. In many cases, simpler and more stable approaches performed better on unseen data.

---

# Reproducibility Instructions

## Environment

The solution was developed and tested in Google Colab using Python 3.11 and an NVIDIA T4 GPU.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Solution

To reproduce the final results and generate predictions:

```bash
python solution.py
```

After execution, the pipeline automatically generates:
- `results.json`
- `predictions.csv`

The generated `predictions.csv` file corresponds to the final submission.

## Important Details

Only the files allowed by the competition rules were modified:
- `aggregation.py`
- `probe.py`
- `splitting.py`

The following infrastructure files were intentionally left unchanged:
- `model.py`
- `evaluate.py`
- `solution.py`

The solution uses hidden-state representations extracted from the `Qwen/Qwen2.5-0.5B` model.

Input features are built from the concatenation of:

```text
prompt + response
```

The final pipeline uses:
- masked mean pooling over valid tokens,
- representations from the final transformer layer,
- a regularized probe classifier,
- stratified 5-fold cross-validation.

---

# Final Solution Description

The final solution is focused on improving the stability of hallucination detection using hidden-state representations extracted from the Qwen2.5-0.5B language model.

Only the components allowed by the competition rules were modified:
- `aggregation.py`
- `probe.py`
- `splitting.py`

The infrastructure files responsible for model loading and evaluation remained unchanged.

## Aggregation Strategy

The original baseline used the hidden representation of the last valid token from the final transformer layer. In practice, this approach turned out to be quite sensitive to specific token positions, and small changes in the split sometimes noticeably affected the metric.

In the final version, masked mean pooling over all valid tokens from the last transformer layer is used instead. Padding tokens are removed with the attention mask before aggregation.

This approach produced more stable representations and reduced variance between folds without increasing feature dimensionality too much.

During development, several more complex aggregation methods were also tested:
- concatenation of multiple transformer layers,
- weighted layer fusion,
- geometric features based on hidden-state statistics.

Some of these experiments gave very strong train metrics, but validation and test AUROC usually became less stable. Multi-layer concatenation especially tended to overfit quickly because of the large feature space. As a result, the final pipeline intentionally stays relatively simple.

## Probe Classifier

The baseline probe showed obvious overfitting. In multiple runs, train AUROC was almost perfect, while validation and test scores stayed much lower.

To improve generalization, the classifier architecture was simplified and regularized. Dropout was added, the hidden layer size was adjusted, and several heavier architectures were discarded during experiments.

At the same time, validation-based threshold tuning was kept because it consistently improved the balance between precision and recall.

The final probe behaves more consistently across different folds and produces noticeably more stable results than earlier versions.

## Data Splitting Strategy

The original baseline relied on a single train/validation/test split. Because of that, the final metric depended heavily on one particular partition of the dataset.

The final solution uses stratified 5-fold cross-validation instead. This made it possible to evaluate the model on several independent splits while preserving class balance in each fold.

Cross-validation also helped separate genuinely useful changes from modifications that only improved performance on one random split. In practice, this turned out to be one of the most important parts of the final pipeline.

---

# Experiments and Failed Attempts

A significant part of the work was spent on experiments that ultimately were not included in the final solution. In most cases, the main issue was overfitting: train metrics improved very quickly, while validation and test AUROC either stayed unchanged or became worse.

## Multi-layer Aggregation

Several experiments used hidden states from multiple transformer layers instead of only the final one.

The following variants were tested:
- concatenation of the last two layers,
- concatenation of the last four layers,
- weighted combinations of layers.

These approaches increased feature dimensionality significantly and caused strong overfitting. In one configuration, the feature dimension increased from 896 to more than 2600 features, while average test AUROC dropped to around 64%.

Because of this, the final solution keeps only the last transformer layer.

## Geometric Features

Additional handcrafted features based on hidden-state geometry were also explored:
- layer activation norms,
- cosine similarity between layers,
- representation drift statistics,
- sequence-level statistics.

The idea itself looked promising, especially for hallucination detection, but in practice the added features mostly increased noise. Validation stability became worse, and average AUROC dropped below the simpler baseline configuration.

For this reason, geometric features were removed from the final version.

## PCA and Dimensionality Reduction

PCA was tested as a way to reduce feature dimensionality before training the probe classifier.

This helped slightly with memory usage and training stability for large feature vectors, but useful information from hidden representations was also lost during compression.

As a result, the final metrics became less stable, so this approach was discarded.

## Larger Probe Architectures

Several deeper and wider probe architectures were tested:
- multiple hidden layers,
- larger hidden dimensions,
- stronger nonlinearities.

These models learned the training set extremely quickly and achieved very high train AUROC. However, validation and test performance usually became worse.

Given the relatively small dataset size, simpler classifiers generalized much better.

## Alternative Pooling Strategies

Different pooling strategies were explored during development:
- max pooling,
- last-token pooling,
- weighted pooling,
- mixed pooling.

Mean pooling over valid tokens consistently produced the most stable behaviour across folds, even if some alternative methods occasionally gave slightly higher scores on individual splits.

In the final solution, stability and reproducibility were prioritized over isolated improvements on single runs.

---

# Summary of Key Experiments

| Experiment | Main Idea | Result |
|---|---|---|
| Original baseline | Last-token representation from final layer | ~72.9 test AUROC |
| Multi-layer concatenation | Concatenation of several transformer layers | Strong overfitting, test AUROC dropped to ~64 |
| Geometric features | Additional handcrafted hidden-state statistics | Validation became less stable, AUROC dropped below ~69 |
| PCA reduction | Dimensionality reduction before probe training | Reduced useful signal, lower test stability |
| Larger probe architectures | Deeper and wider classifiers | Very high train AUROC but poor generalization |
| Alternative pooling methods | Max pooling and weighted pooling | Less stable across folds than mean pooling |
| Final solution | Mean pooling + regularized probe + 5-fold CV | ~72.8 average test AUROC |

---

# Final Remarks

The final solution prioritizes stability and generalization over architectural complexity.

Many experimental configurations achieved extremely high train AUROC, especially when using larger feature spaces or deeper probe architectures. However, these approaches usually generalized poorly and produced unstable validation results.

The best overall performance was achieved with a relatively simple setup:
- final-layer hidden representations,
- masked mean pooling,
- regularized probe training,
- stratified 5-fold cross-validation.

The final pipeline remained lightweight, reproducible, and fully compatible with the original competition infrastructure while achieving stable performance across folds.