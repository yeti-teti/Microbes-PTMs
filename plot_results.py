import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the saved results file
results_path = os.path.join("results", "results.pkl")
with open(results_path, "rb") as f:
    results = pickle.load(f)

# Extract the history, predictions, and actuals
history = results['history']
predictions = results['predictions']
actuals = results['actuals']

# Ensure history is a pandas DataFrame (if not, convert it)
if not isinstance(history, pd.DataFrame):
    history = pd.DataFrame(history)

# Create plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot Training and Validation Loss
ax1.plot(history['epoch'], history['train_loss'], label='Training Loss')
ax1.plot(history['epoch'], history['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()

# Plot Training and Validation Correlation
ax2.plot(history['epoch'], history['train_correlation'], label='Training Correlation')
ax2.plot(history['epoch'], history['val_correlation'], label='Validation Correlation')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Correlation')
ax2.set_title('Training and Validation Correlation')
ax2.legend()

# Plot Predictions vs. Actuals
ax3.scatter(actuals, predictions, alpha=0.1)
ax3.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
ax3.set_xlabel('Actual Values')
ax3.set_ylabel('Predicted Values')
corr = np.corrcoef(actuals, predictions)[0, 1]
ax3.set_title(f'Predictions vs Actuals\nr={corr:.3f}')

plt.tight_layout()
plt.show()
