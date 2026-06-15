import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

output_root = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots"

# ============================================
# Generate synthetic data that mimics real pattern
# ============================================
np.random.seed(42)
n_samples = 1000

# Generate radiomics and pathomics risk scores
radio_risk = np.random.uniform(0, 1, n_samples)
patho_risk = np.random.uniform(0, 1, n_samples)

# True label follows "trust the confident expert" rule
# High risk if EITHER expert is confident and correct
true_label = np.zeros(n_samples)

# Region 1: High radio, low patho → high risk (radio is right)
mask1 = (radio_risk > 0.7) & (patho_risk < 0.3)
true_label[mask1] = 1

# Region 2: Low radio, high patho → high risk (patho is right)
mask2 = (radio_risk < 0.3) & (patho_risk > 0.7)
true_label[mask2] = 1

# Region 3: Both high → high risk
mask3 = (radio_risk > 0.7) & (patho_risk > 0.7)
true_label[mask3] = 1

# Region 4: Both low → low risk (stays 0)

# Add some noise (mislabeled cases)
noise_mask = np.random.random(n_samples) < 0.05
true_label[noise_mask] = 1 - true_label[noise_mask]

# Create feature matrix
X = np.column_stack([radio_risk, patho_risk])

# ============================================
# Train models
# ============================================
# Weighted averaging (Logistic Regression on the two risks)
lr = LogisticRegression()
lr.fit(X, true_label)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X, true_label)

# ============================================
# Create decision boundary visualization
# ============================================
xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Get predictions
lr_pred = lr.predict_proba(grid)[:, 1].reshape(xx.shape)
rf_pred = rf.predict_proba(grid)[:, 1].reshape(xx.shape)

# ============================================
# Plot side-by-side comparison
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Logistic Regression (Weighted Averaging)
ax1 = axes[0]
contour1 = ax1.contourf(xx, yy, lr_pred, levels=20, cmap='RdBu', alpha=0.8)
ax1.scatter(radio_risk[true_label==1], patho_risk[true_label==1], 
            c='red', marker='o', s=20, alpha=0.5, label='Event')
ax1.scatter(radio_risk[true_label==0], patho_risk[true_label==0], 
            c='blue', marker='s', s=20, alpha=0.5, label='Censored')
ax1.set_xlabel('Radiomics Risk Score', fontsize=12)
ax1.set_ylabel('Pathomics Risk Score', fontsize=12)
ax1.set_title('Weighted Averaging (Logistic Regression)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Draw the problem regions
ax1.add_patch(plt.Rectangle((0.7, 0), 0.3, 0.3, 
                             fill=False, edgecolor='black', linestyle='--', 
                             linewidth=2, label='Disagreement Region'))
ax1.add_patch(plt.Rectangle((0, 0.7), 0.3, 0.3, 
                             fill=False, edgecolor='black', linestyle='--', 
                             linewidth=2))

# Plot 2: Random Forest
ax2 = axes[1]
contour2 = ax2.contourf(xx, yy, rf_pred, levels=20, cmap='RdBu', alpha=0.8)
ax2.scatter(radio_risk[true_label==1], patho_risk[true_label==1], 
            c='red', marker='o', s=20, alpha=0.5, label='Event')
ax2.scatter(radio_risk[true_label==0], patho_risk[true_label==0], 
            c='blue', marker='s', s=20, alpha=0.5, label='Censored')
ax2.set_xlabel('Radiomics Risk Score', fontsize=12)
ax2.set_ylabel('Pathomics Risk Score', fontsize=12)
ax2.set_title('Random Forest', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Draw the problem regions
ax2.add_patch(plt.Rectangle((0.7, 0), 0.3, 0.3, 
                             fill=False, edgecolor='black', linestyle='--', 
                             linewidth=2))
ax2.add_patch(plt.Rectangle((0, 0.7), 0.3, 0.3, 
                             fill=False, edgecolor='black', linestyle='--', 
                             linewidth=2))

plt.tight_layout()
plt.savefig(f'{output_root}/rf_vs_weighting_comparison.png', dpi=150)
plt.show()

# ============================================
# Print performance metrics
# ============================================
from sklearn.metrics import accuracy_score, f1_score

y_pred_lr = lr.predict(X)
y_pred_rf = rf.predict(X)

print("\n" + "="*60)
print("Performance Comparison on Disagreement Regions")
print("="*60)

# Evaluate on the problematic regions (where experts disagree strongly)
disagree_mask = ((radio_risk > 0.7) & (patho_risk < 0.3)) | ((radio_risk < 0.3) & (patho_risk > 0.7))

if disagree_mask.sum() > 0:
    acc_lr_disagree = accuracy_score(true_label[disagree_mask], y_pred_lr[disagree_mask])
    acc_rf_disagree = accuracy_score(true_label[disagree_mask], y_pred_rf[disagree_mask])
    
    print(f"\nDisagreement regions (high confidence split):")
    print(f"  Weighted Averaging accuracy: {acc_lr_disagree:.3f}")
    print(f"  Random Forest accuracy:      {acc_rf_disagree:.3f}")
    print(f"  Improvement:                 {acc_rf_disagree - acc_lr_disagree:+.3f}")

print("\n" + "="*60)