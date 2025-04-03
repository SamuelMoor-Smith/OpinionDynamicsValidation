import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data
labels = ['imwbcnt-UK', 'stfdem-FI', 'trstun-SI', 'ppltrst-HU', 'happy-PI', 'rlgdgr-PI']
models = ['Deffuant', 'HK Averaging', 'Carpentras', 'Duggins']

zero = [0.3483786152, 0.4771633051, 0.3058933583, 0.3265771812, 0.3006647673, 0.3013358779]
deffuant = [0.3492498927, 0.4772413793, 0.3097099199, 0.3274307812, 0.30735652, 0.3032951404]
hk = [0.3485735618, 0.4776997471, 0.3183971259, 0.3298483951, 0.3016968279, 0.3014374477]
carpentras = [0.3516377365, 0.4771478658, 0.3045434178, 0.3197907786, 0.3047125542, 0.2951609512]
duggins = [0.3501342154, 0.4750169645, 0.3073133609, 0.3275180421, 0.3007340417, 0.3119763147]

model_data = {
    'Deffuant': deffuant,
    'HK Averaging': hk,
    'Carpentras': carpentras,
    'Duggins': duggins
}

# Create a noisy version of each value to simulate distributions
data = []
samples_per_point = 50  # Number of noisy samples per data point
noise_std = 0.005        # Adjust for more or less spread

for i, label in enumerate(labels):
    for model_name in models:
        baseline = zero[i]
        model_val = model_data[model_name][i]
        rel_loss = (baseline - model_val) / baseline
        noisy_vals = rel_loss + np.random.normal(0, noise_std, samples_per_point)
        for val in noisy_vals:
            data.append([label, model_name, val])

df = pd.DataFrame(data, columns=["Dataset", "Model", "Scaled Loss"])

# Plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="Dataset", y="Scaled Loss", hue="Model", split=True, inner="quartile")

plt.title('Model Performance Comparison (Violin Plot with Noise)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("model_comparison_violin_noisy_split.png", dpi=300, bbox_inches="tight")
# plt.show()
