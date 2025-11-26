import neat
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, PrecisionRecallDisplay, 
    roc_auc_score, roc_curve, auc
)

# ============================================================================
# LOAD DATA (Same as training)
# ============================================================================

print("Loading data...")
df = pd.read_csv("Data/StressAppraisal.csv")
data = df.drop(columns=["Productivity", "Mood", "Stress_Numeric", "Stress"])
labels = df["Stress"]
labels_encoded, uniques = pd.factorize(labels)

# Same split as training
data_train, data_test, labels_train, labels_test = train_test_split(
    data, labels_encoded, test_size=0.2, random_state=17, stratify=labels_encoded
)

# Binarize labels for multi-class metrics
labels_test_bin = label_binarize(labels_test, classes=range(len(uniques)))

print(f"Test set: {len(labels_test)} samples")
print(f"Classes: {list(uniques)}")


# ============================================================================
# LOAD NEAT MODEL AND SCALER
# ============================================================================

print("\nLoading NEAT model...")

# Load the saved genome
with open('final_genome.pkl', 'rb') as f:  # or 'final_genome.pkl'
    genome = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load config
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'neat_config'
)

# Create network from genome
net = neat.nn.FeedForwardNetwork.create(genome, config)

print(f"✓ Model loaded: {len(genome.nodes)} nodes, {len(genome.connections)} connections")
print(f"  Training fitness: {genome.fitness:.4f}")


# ============================================================================
# PREPROCESS TEST DATA
# ============================================================================

print("\nPreprocessing test data...")
data_test_scaled = scaler.transform(data_test.values)


# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("Generating predictions...")

# Get predictions and probability estimates
labels_pred = []
labels_pred_proba = []

for xi in data_test_scaled:
    output = net.activate(xi)  # Raw network output (4 values)
    
    # Convert to probabilities using softmax
    exp_output = np.exp(output - np.max(output))  # Subtract max for numerical stability
    proba = exp_output / exp_output.sum()
    
    labels_pred_proba.append(proba)
    labels_pred.append(np.argmax(proba))

labels_pred = np.array(labels_pred)
labels_pred_proba = np.array(labels_pred_proba)

print(f"✓ Generated predictions for {len(labels_pred)} samples")


# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================

print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(labels_test, labels_pred, target_names=uniques))


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(labels_test, labels_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=uniques)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap="Blues", values_format="d", ax=ax)
ax.set_title("Confusion Matrix (NEAT Model)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('neat_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: neat_confusion_matrix.png")
plt.show()


# ============================================================================
# PRECISION-RECALL CURVES (One-vs-All)
# ============================================================================

print("\nGenerating precision-recall curves...")

fig, ax = plt.subplots(figsize=(10, 8))

for i in range(len(uniques)):
    precision, recall, _ = precision_recall_curve(
        labels_test_bin[:, i], 
        labels_pred_proba[:, i]
    )
    
    # Calculate average precision
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(labels_test_bin[:, i], labels_pred_proba[:, i])
    
    # Plot
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot(ax=ax, name=f"{uniques[i]} (AP={ap:.3f})", linewidth=2)

ax.set_title("Precision-Recall Curves (One-vs-All)", fontsize=16, fontweight='bold')
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('neat_precision_recall_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: neat_precision_recall_curves.png")
plt.show()


# ============================================================================
# AUROC SCORES
# ============================================================================

print("\n" + "="*70)
print("AUROC SCORES")
print("="*70)

# Macro-averaged AUROC (One-vs-Rest)
try:
    auroc_macro_ovr = roc_auc_score(
        labels_test, 
        labels_pred_proba, 
        multi_class='ovr', 
        average='macro'
    )
    print(f"Multiclass AUROC (OVR, Macro):     {auroc_macro_ovr:.4f}")
except Exception as e:
    print(f"Could not calculate macro AUROC: {e}")

# Weighted AUROC (One-vs-One)
try:
    auroc_weighted_ovo = roc_auc_score(
        labels_test, 
        labels_pred_proba, 
        multi_class='ovo', 
        average='weighted'
    )
    print(f"Multiclass AUROC (OVO, Weighted):  {auroc_weighted_ovo:.4f}")
except Exception as e:
    print(f"Could not calculate weighted AUROC: {e}")

# Per-class AUROC
print("\nPer-class AUROC (One-vs-All):")
for i in range(len(uniques)):
    try:
        auroc_class = roc_auc_score(
            labels_test_bin[:, i], 
            labels_pred_proba[:, i]
        )
        print(f"  {uniques[i]:30s}: {auroc_class:.4f}")
    except Exception as e:
        print(f"  {uniques[i]:30s}: Could not calculate ({e})")


# ============================================================================
# ROC CURVES (One-vs-All)
# ============================================================================

print("\nGenerating ROC curves...")

false_positive_rate = dict()
true_positive_rate = dict()
roc_auc = dict()

# Calculate ROC curve and AUC for each class
for i in range(len(uniques)):
    false_positive_rate[i], true_positive_rate[i], _ = roc_curve(
        labels_test_bin[:, i], 
        labels_pred_proba[:, i]
    )
    roc_auc[i] = auc(false_positive_rate[i], true_positive_rate[i])

# Plot ROC curves
fig, ax = plt.subplots(figsize=(10, 8))

for i in range(len(uniques)):
    ax.plot(
        false_positive_rate[i], 
        true_positive_rate[i],
        linewidth=2,
        label=f'{uniques[i]} (AUC = {roc_auc[i]:.3f})'
    )

# Plot diagonal (random classifier)
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves (One-vs-All)', fontsize=16, fontweight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('neat_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: neat_roc_curves.png")
plt.show()


# ============================================================================
# ADDITIONAL METRICS
# ============================================================================

print("\n" + "="*70)
print("ADDITIONAL METRICS")
print("="*70)

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

accuracy = accuracy_score(labels_test, labels_pred)
balanced_acc = balanced_accuracy_score(labels_test, labels_pred)
f1_macro = f1_score(labels_test, labels_pred, average='macro')
f1_weighted = f1_score(labels_test, labels_pred, average='weighted')

print(f"Accuracy:                {accuracy:.4f}")
print(f"Balanced Accuracy:       {balanced_acc:.4f}")
print(f"F1 Score (Macro):        {f1_macro:.4f}")
print(f"F1 Score (Weighted):     {f1_weighted:.4f}")

# Per-class F1 scores
print("\nPer-class F1 Scores:")
f1_per_class = f1_score(labels_test, labels_pred, average=None)
for i, (name, score) in enumerate(zip(uniques, f1_per_class)):
    print(f"  {name:30s}: {score:.4f}")


# ============================================================================
# PREDICTION DISTRIBUTION
# ============================================================================

print("\n" + "="*70)
print("PREDICTION DISTRIBUTION")
print("="*70)

# True distribution
print("\nTrue label distribution:")
unique_true, counts_true = np.unique(labels_test, return_counts=True)
for cls, count in zip(unique_true, counts_true):
    pct = count / len(labels_test) * 100
    print(f"  {uniques[cls]:30s}: {count:3d} ({pct:.1f}%)")

# Predicted distribution
print("\nPredicted label distribution:")
unique_pred, counts_pred = np.unique(labels_pred, return_counts=True)
pred_dict = dict(zip(unique_pred, counts_pred))
for i in range(len(uniques)):
    count = pred_dict.get(i, 0)
    pct = count / len(labels_pred) * 100 if len(labels_pred) > 0 else 0
    print(f"  {uniques[i]:30s}: {count:3d} ({pct:.1f}%)")


# ============================================================================
# CONFIDENCE ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("CONFIDENCE ANALYSIS")
print("="*70)

# Calculate confidence (max probability) for each prediction
confidences = np.max(labels_pred_proba, axis=1)

print(f"\nPrediction confidence statistics:")
print(f"  Mean confidence:   {confidences.mean():.4f}")
print(f"  Median confidence: {np.median(confidences):.4f}")
print(f"  Min confidence:    {confidences.min():.4f}")
print(f"  Max confidence:    {confidences.max():.4f}")

# Confidence by correctness
correct_mask = labels_pred == labels_test
correct_confidences = confidences[correct_mask]
incorrect_confidences = confidences[~correct_mask]

print(f"\nConfidence for correct predictions:   {correct_confidences.mean():.4f}")
print(f"Confidence for incorrect predictions: {incorrect_confidences.mean():.4f}")

# Plot confidence distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green')
axes[0].hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red')
axes[0].set_xlabel('Confidence', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot
axes[1].boxplot([correct_confidences, incorrect_confidences], 
                labels=['Correct', 'Incorrect'],
                patch_artist=True)
axes[1].set_ylabel('Confidence', fontsize=12)
axes[1].set_title('Confidence by Correctness', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('neat_confidence_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: neat_confidence_analysis.png")
plt.show()


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  1. neat_confusion_matrix.png")
print("  2. neat_precision_recall_curves.png")
print("  3. neat_roc_curves.png")
print("  4. neat_confidence_analysis.png")
print("\nKey Metrics:")
print(f"  Accuracy:        {accuracy:.4f}")
print(f"  F1 (Macro):      {f1_macro:.4f}")
print(f"  AUROC (Macro):   {auroc_macro_ovr:.4f}")
print(f"  Avg Confidence:  {confidences.mean():.4f}")
print("="*70)