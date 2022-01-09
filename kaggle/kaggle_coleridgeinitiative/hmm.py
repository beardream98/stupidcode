import numpy as np
def compute_fbeta(y_true,
                  y_pred,
                  beta=0.5) :
    """Compute the Jaccard-based micro FBeta score.

    References
    ----------
    - https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/overview/evaluation
    """

    def _jaccard_similarity(str1: str, str2: str) -> float:
        a = set(str1.split()) 
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    tp = 0  # true positive
    fp = 0  # false positive
    fn = 0  # false negative
    for ground_truth_list, predicted_string_list in zip(y_true, y_pred):
        predicted_string_list_sorted = sorted(predicted_string_list)
        for ground_truth in sorted(ground_truth_list):
            if len(predicted_string_list_sorted) == 0:
                fn += 1
            else:
                similarity_scores = [
                    _jaccard_similarity(ground_truth, predicted_string)
                    for predicted_string in predicted_string_list_sorted
                ]
                matched_idx = np.argmax(similarity_scores)
                if similarity_scores[matched_idx] >= 0.5:
                    predicted_string_list_sorted.pop(matched_idx)
                    tp += 1
                else:
                    fn += 1
        fp += len(predicted_string_list_sorted)

    tp *= (1 + beta ** 2)
    fn *= beta ** 2
    fbeta_score = tp / (tp + fp + fn)
    return fbeta_score

a=['adni|alzheimer s disease neuroimaging initiative adni',
 'trends in international mathematics and science study|nces common core of data|common core of data',
 'sea lake and overland surges from hurricanes|slosh model|noaa storm surge inundation',
 'rural urban continuum codes']

b=['alzheimer s disease neuroimaging initiative adni',
 'trends in international mathematics and science study',
 'slosh model',
 'rural urban continuum codes']

a=[ string.split("|") for string in a]
b=[ string.split("|") for string in b]
print(compute_fbeta(b,a))