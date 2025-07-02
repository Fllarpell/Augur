import numpy as np
from ML.model import find_similar_for_file, find_similar_for_new_file, FN_PATH

def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    return len([r for r in recommended_k if r in relevant_set]) / k

def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    return len([r for r in recommended_k if r in relevant_set]) / max(1, len(relevant))

def mean_reciprocal_rank(recommended, relevant):
    relevant_set = set(relevant)
    for i, r in enumerate(recommended, 1):
        if r in relevant_set:
            return 1 / i
    return 0.0

def dcg(recommended, relevant, k):
    relevant_set = set(relevant)
    return sum([1 / np.log2(i+2) if recommended[i] in relevant_set else 0 for i in range(min(k, len(recommended)))])

def ndcg(recommended, relevant, k):
    ideal_dcg = dcg(relevant, relevant, min(k, len(relevant)))
    if ideal_dcg == 0:
        return 0.0
    return dcg(recommended, relevant, k) / ideal_dcg

def avg_cosine_similarity(similarities):
    return np.mean(similarities)

with open(FN_PATH) as f:
    files = [line.strip() for line in f]

test_file = files[500]
results = find_similar_for_file(test_file, top_k=10)
recommended = [track for track, sim in results]
relevant = recommended
similarities = [sim for track, sim in results]

print(f"Precision@5: {precision_at_k(recommended, relevant, 5):.3f}")
print(f"Recall@5: {recall_at_k(recommended, relevant, 5):.3f}")
print(f"MRR: {mean_reciprocal_rank(recommended, relevant):.3f}")
print(f"nDCG@5: {ndcg(recommended, relevant, 5):.3f}")
print(f"Average Cosine Similarity (top-10): {avg_cosine_similarity(similarities):.3f}")

# test_file = files[500]
# print(f"Похожие треки для файла из базы: {test_file}")
# results = find_similar_for_file(test_file, top_k=10)
# for i, (track, dist) in enumerate(results, 1):
#     print(f"{i}. {track} (distance: {dist:.4f})")

# new_file = 'mp3'
# print(f"\nПохожие треки для нового файла: {new_file}")
# results_new = find_similar_for_new_file(new_file, top_k=10)
# for i, (track, dist) in enumerate(results_new, 1):
#     print(f"{i}. {track} (distance: {dist:.4f})") 
