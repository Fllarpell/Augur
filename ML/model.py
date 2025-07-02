import os
import numpy as np
from tqdm import tqdm
import multiprocessing
import faiss
import librosa
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tensorflow as tf
import soundfile as sf

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../fma_large'))
EMB_PATH = os.path.join(os.path.dirname(__file__), "embeddings_yamnet.npy")
FN_PATH  = os.path.join(os.path.dirname(__file__), "filenames_yamnet.txt")
BATCH_SIZE = 128
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

EMBEDDING_SIZE = 1024

def load_yamnet_model():
    import tensorflow_hub as hub
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    return model

yamnet_model = load_yamnet_model()

def find_mp3_files(root_dir):
    mp3_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                mp3_files.append(os.path.join(root, file))
    return mp3_files

def load_and_embed(fn, embedding_size=None):
    audio, sr = sf.read(fn)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    scores, embeddings, spectrogram = yamnet_model(audio)
    emb_mean = embeddings.numpy().mean(axis=0)
    
    emb_norm = emb_mean / (np.linalg.norm(emb_mean) + 1e-8)
    return fn, emb_norm

def extract_batch(filenames, embedding_size=None, batch_size=BATCH_SIZE):
    results = []
    for fn in filenames:
        try:
            results.append(load_and_embed(fn))
        except Exception as e:
            print(f"Error processing {fn}: {e}")
    return results

def build_or_load_embeddings(mp3_files):
    if os.path.exists(EMB_PATH) and os.path.exists(FN_PATH):
        print("Loading existing embeddings...")
        embeddings = np.load(EMB_PATH)
        with open(FN_PATH) as f:
            fnames = [line.strip() for line in f]
        return embeddings, fnames

    all_embs = []
    all_fns  = []
    errors   = []
    batches = [mp3_files[i:i+BATCH_SIZE] for i in range(0, len(mp3_files), BATCH_SIZE)]
    for batch in tqdm(batches, desc="Batch feature extraction"):
        try:
            extracted = extract_batch(batch)
            for fn, emb in extracted:
                all_fns.append(fn)
                all_embs.append(emb)
        except Exception as e:
            for fn in batch:
                errors.append((fn, str(e)))

    all_embs = np.stack(all_embs).astype('float32')
    np.save(EMB_PATH, all_embs)
    with open(FN_PATH, "w") as f:
        f.write("\n".join(all_fns))
    if errors:
        with open("errors_yamnet.txt", "w") as f:
            for fn, err in errors:
                f.write(f"{fn}: {err}\n")
        print(f"Errors for {len(errors)} files; see errors_yamnet.txt")
    return all_embs, all_fns

def find_similar_tracks(embeddings, filenames, top_k=10):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    D, I = index.search(embeddings, top_k+1)
    results = []
    for i in range(len(filenames)):
        sims = []
        for dist, idx in zip(D[i], I[i]):
            if idx == i:
                continue
            sims.append((filenames[idx], float(dist)))
            if len(sims) >= top_k:
                break
        results.append((filenames[i], sims))
    return results

def find_similar_for_file(query_file, top_k=10):
    if not os.path.exists(EMB_PATH) or not os.path.exists(FN_PATH):
        raise RuntimeError("Embeddings or filenames not found. Please run embedding extraction first.")
    embeddings = np.load(EMB_PATH)
    with open(FN_PATH) as f:
        filenames = [line.strip() for line in f]
    try:
        idx = filenames.index(os.path.abspath(query_file))
    except ValueError:
        raise ValueError(f"File {query_file} not found in precomputed filenames.")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    D, I = index.search(embeddings[idx:idx+1], top_k+1)
    sims = []
    for dist, i in zip(D[0], I[0]):
        if i == idx:
            continue
        sims.append((filenames[i], float(dist)))
        if len(sims) >= top_k:
            break
    return sims

def find_similar_for_new_file(query_file, top_k=10):
    if not os.path.exists(EMB_PATH) or not os.path.exists(FN_PATH):
        raise RuntimeError("Embeddings or filenames not found. Please run embedding extraction first.")
    embeddings = np.load(EMB_PATH)
    with open(FN_PATH) as f:
        filenames = [line.strip() for line in f]
    
    audio, sr = sf.read(query_file)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    scores, embeddings_new, spectrogram = yamnet_model(audio)
    emb_mean = embeddings_new.numpy().mean(axis=0)
    emb_norm = emb_mean / (np.linalg.norm(emb_mean) + 1e-8)
    emb_norm = emb_norm.astype('float32').reshape(1, -1)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    D, I = index.search(emb_norm, top_k)
    sims = []
    for dist, i in zip(D[0], I[0]):
        sims.append((filenames[i], float(dist)))
    return sims

if __name__ == "__main__":
    mp3s = find_mp3_files(DATA_DIR)
    embeddings, fnames = build_or_load_embeddings(mp3s)
    sims = find_similar_tracks(embeddings, fnames, top_k=10)
    with open("similar_tracks_yamnet.txt", "w") as f:
        for track, sim_list in sims:
            f.write(f"Track: {track}\n")
            for fn_sim, dist in sim_list:
                f.write(f"    {fn_sim} (distance: {dist:.4f})\n")
            f.write("\n")
