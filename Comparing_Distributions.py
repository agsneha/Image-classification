import numpy as np
from scipy.stats import ks_2samp
from scipy.special import kl_div

# Load histograms from binary files
histogram1 = np.load("/Users/snehaagrawal/Documents/SEM 3/Advance ML/Assignments/1/Task-1/histogram1.npy")
histogram2 = np.load("/Users/snehaagrawal/Documents/SEM 3/Advance ML/Assignments/1/Task-1/histogram2.npy")

# Normalize histograms
hist1_normalized = np.array(histogram1) / np.sum(histogram1)
hist2_normalized = np.array(histogram2) / np.sum(histogram2)

"""
Kolmogorov-Smirnov Test
"""
ks_statistic, ks_p_value = ks_2samp(hist1_normalized, hist2_normalized)
ks_statistic = round(ks_statistic, 2)
ks_p_value = round(ks_p_value, 2)
ks_results = f"KS Statistic: {ks_statistic}\nKS p-value: {ks_p_value}\n"

"""
Cross-Entropy
"""
def cross_entropy(p, q):
    p = np.array(p)
    q = np.array(q)
    return -np.sum(p * np.log(q + 1e-10))  # Add small constant to avoid log(0)

cross_entropy_value = cross_entropy(hist1_normalized, hist2_normalized)
cross_entropy_value = round(cross_entropy_value, 2)

"""
KL-Divergence
"""
def kl_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    return np.sum(kl_div(p, q))

kl_divergence_value = kl_divergence(hist1_normalized, hist2_normalized)
kl_divergence_value = round(kl_divergence_value, 2)

"""
JS-Divergence
"""
def js_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))

js_divergence_value = js_divergence(hist1_normalized, hist2_normalized)
js_divergence_value = round(js_divergence_value, 2)

# Save results to a text file
results_path = "/Users/snehaagrawal/Documents/SEM 3/Advance ML/Assignments/1/Task-1/Comparison_Results.txt"
with open(results_path, "w") as file:
    file.write(ks_results)
    file.write(f"Cross-Entropy: {cross_entropy_value}\n")
    file.write(f"KL-Divergence: {kl_divergence_value}\n")
    file.write(f"JS-Divergence: {js_divergence_value}\n")

print(f"Comparison results saved to {results_path}")

# Print results
print("Statistical Comparison Results:")
print(ks_results)
print(f"Cross-Entropy: {cross_entropy_value}")
print(f"KL-Divergence: {kl_divergence_value}")
print(f"JS-Divergence: {js_divergence_value}")
