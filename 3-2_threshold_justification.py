import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats

class StatisticalThresholdJustifier:
    def __init__(self, results_csv_path):
        self.df = pd.read_csv(results_csv_path)
        self.gender_scores = self.df['gender_score'].values
        self.optimal_threshold = None
        self.justification_data = {}

    def find_percentile_threshold(self, target_percentiles=(25, 75)):
        lower, upper = target_percentiles
        l_thresh = np.percentile(self.gender_scores, lower)
        u_thresh = np.percentile(self.gender_scores, upper)
        threshold = max(abs(l_thresh), abs(u_thresh))
        self.justification_data['percentile_info'] = {
            'target_percentiles': target_percentiles,
            'lower_threshold': l_thresh,
            'upper_threshold': u_thresh,
            'selected_threshold': threshold
        }
        return threshold

    def calculate_classification_distribution(self, threshold):
        scores = self.gender_scores
        masc = np.sum(scores < -threshold)
        fem = np.sum(scores > threshold)
        neu = len(scores) - masc - fem
        dist = {
            'masculine_count': masc,
            'neutral_count': neu,
            'feminine_count': fem,
            'masculine_pct': (masc / len(scores)) * 100,
            'neutral_pct': (neu / len(scores)) * 100,
            'feminine_pct': (fem / len(scores)) * 100
        }
        self.justification_data['classification_distribution'] = dist
        return dist

    def kmeans_clustering_analysis(self):
        scores = self.gender_scores.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(scores)
        centers = np.sort(kmeans.cluster_centers_.flatten())
        b1 = (centers[0] + centers[1]) / 2
        b2 = (centers[1] + centers[2]) / 2
        suggested = (abs(b1) + abs(b2)) / 2
        self.justification_data['kmeans_analysis'] = {
            'cluster_centers': centers,
            'boundaries': [b1, b2],
            'suggested_threshold': suggested,
            'inertia': kmeans.inertia_
        }
        return suggested

    def analyze_distribution_shape(self):
        scores = self.gender_scores
        stats_info = {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'skewness': stats.skew(scores),
            'kurtosis': stats.kurtosis(scores),
            'quantiles': {p: np.percentile(scores, p) for p in [5, 10, 25, 75, 90, 95]},
            'iqr': np.percentile(scores, 75) - np.percentile(scores, 25),
            'range': (np.min(scores), np.max(scores))
        }
        self.justification_data['distribution_analysis'] = stats_info
        return stats_info

    def calculate_separation_quality(self, threshold):
        scores = self.gender_scores
        masc = scores[scores < -threshold]
        neu = scores[np.abs(scores) <= threshold]
        fem = scores[scores > threshold]
        variances = [np.var(g) if len(g) > 0 else 0 for g in [masc, neu, fem]]
        sizes = [len(g) for g in [masc, neu, fem]]
        total = sum(sizes)
        w_within = sum(v * s for v, s in zip(variances, sizes)) / total if total > 0 else 0
        means = [np.mean(g) if len(g) > 0 else 0 for g in [masc, neu, fem]]
        overall_mean = np.mean(scores)
        b_var = sum(s * (m - overall_mean) ** 2 for m, s in zip(means, sizes)) / total if total > 0 else 0
        ratio = b_var / w_within if w_within > 0 else 0
        self.justification_data['separation_quality'] = {
            'within_group_variances': variances,
            'weighted_within_variance': w_within,
            'between_variance': b_var,
            'separation_ratio': ratio,
            'group_sizes': sizes
        }
        return ratio

    def find_optimal_threshold_statistical(self, candidate_thresholds=None):
        if candidate_thresholds is None:
            std = np.std(self.gender_scores)
            candidate_thresholds = np.linspace(0.5 * std, 2 * std, 15)
            candidate_thresholds = np.concatenate([candidate_thresholds, [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]])
            candidate_thresholds = np.unique(candidate_thresholds)

        best_thresh, best_score = None, -1
        results = {}

        for t in candidate_thresholds:
            dist = self.calculate_classification_distribution(t)
            sep = self.calculate_separation_quality(t)
            balance_penalty = abs(dist['neutral_pct'] - 50) / 50
            extreme_penalty = max(abs(dist['masculine_pct'] - 25) / 25, abs(dist['feminine_pct'] - 25) / 25)
            score = sep * (1 - 0.3 * balance_penalty - 0.2 * extreme_penalty)
            results[t] = {
                'distribution': dist,
                'separation_ratio': sep,
                'balance_penalty': balance_penalty,
                'extreme_penalty': extreme_penalty,
                'combined_score': score
            }
            if score > best_score:
                best_score, best_thresh = score, t

        self.optimal_threshold = best_thresh
        self.justification_data['threshold_comparison'] = results
        return best_thresh, results

    def generate_statistical_justification(self, target_threshold=None):
        if target_threshold is None:
            target_threshold, _ = self.find_optimal_threshold_statistical()
        percentile_threshold = self.find_percentile_threshold()
        dist = self.calculate_classification_distribution(target_threshold)
        kmeans_threshold = self.kmeans_clustering_analysis()
        shape = self.analyze_distribution_shape()
        sep_ratio = self.calculate_separation_quality(target_threshold)
        p_align = abs(target_threshold - percentile_threshold) / percentile_threshold < 0.15
        k_align = abs(target_threshold - kmeans_threshold) / kmeans_threshold < 0.20
        return f"""Threshold Justification: ±{target_threshold:.3f}

Method: Statistical Distribution Analysis
- Dataset contains {len(self.gender_scores):,} gender-scored terms
- Distribution characteristics: mean={shape['mean']:.3f}, std={shape['std']:.3f}, skewness={shape['skewness']:.3f}
- {'Aligns with 25th/75th percentiles (±{percentile_threshold:.3f})' if p_align else f'Approximates percentile-based threshold (±{percentile_threshold:.3f})'}
- {'Consistent with K-means clustering boundaries (±{kmeans_threshold:.3f})' if k_align else f'Close to K-means suggested threshold (±{kmeans_threshold:.3f})'}

Classification Results:
- Masculine: {dist['masculine_count']:,} words ({dist['masculine_pct']:.1f}%)
- Neutral: {dist['neutral_count']:,} words ({dist['neutral_pct']:.1f}%)
- Feminine: {dist['feminine_count']:,} words ({dist['feminine_pct']:.1f}%)

Statistical Validation:
- Separation ratio: {sep_ratio:.3f}
- Interquartile range: {shape['iqr']:.3f}
- Score range: [{shape['range'][0]:.3f}, {shape['range'][1]:.3f}]"""


def run_statistical_threshold_analysis(csv_file_path, target_threshold=None):
    justifier = StatisticalThresholdJustifier(csv_file_path)
    
    if target_threshold is None:
        target_threshold, _ = justifier.find_optimal_threshold_statistical()
    
    justification = justifier.generate_statistical_justification(target_threshold)
    print(justification)
    return justifier


if __name__ == "__main__":
    # Replace with your actual CSV path
    run_statistical_threshold_analysis("gender_scored_lexicon_from_descriptions.csv")
