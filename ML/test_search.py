from model import find_similar_for_file, find_similar_for_new_file, FN_PATH

with open(FN_PATH) as f:
    files = [line.strip() for line in f]

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
