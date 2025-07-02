import sys
import os
import streamlit as st
from tempfile import NamedTemporaryFile
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ML import model
from ML.test_search import precision_at_k, recall_at_k, mean_reciprocal_rank, ndcg, avg_cosine_similarity

st.title("Augur")

with st.expander("ℹ️ Metrics explanation"):
    st.markdown("""
    **Offline metrics:**
    - **Precision@K**: Fraction of recommended tracks in top-K that are relevant (same genre)
    - **Recall@K**: Fraction of all relevant tracks (same genre) that are in top-K recommendations
    - **nDCG@K**: Ranking quality, higher if relevant tracks are ranked higher
    - **MRR**: Reciprocal of the rank of the first relevant result
    - **Cosine Similarity**: Mean and std of similarity scores for top-K
    
    **Ground truth:**
    - For each test track, all tracks of the same genre (from FMA metadata)
    """)

@st.cache_data(show_spinner=True)
def get_fma_metadata():
    tracks = pd.read_csv('fma_metadata/tracks.csv', header=[0, 1], index_col=0, low_memory=False)
    return tracks

def get_existing_ids_and_paths():
    with open('ML/filenames_yamnet.txt') as f:
        paths = [line.strip() for line in f]
    ids = set(os.path.splitext(os.path.basename(p))[0] for p in paths)
    path_map = {os.path.splitext(os.path.basename(p))[0]: p for p in paths}
    return ids, path_map

def build_genre_ground_truth(tracks, existing_ids):
    track_to_genre = {}
    genre_to_tracks = {}
    for track_id in tracks.index:
        tid = str(track_id).zfill(6)
        if tid in existing_ids:
            genre = tracks.loc[track_id, ('track', 'genre_top')]
            if pd.notnull(genre):
                track_to_genre[tid] = genre
                genre_to_tracks.setdefault(genre, []).append(tid)
    return track_to_genre, genre_to_tracks

tracks = get_fma_metadata()
existing_ids, id_to_path = get_existing_ids_and_paths()
track_to_genre, genre_to_tracks = build_genre_ground_truth(tracks, existing_ids)

top_n = 10
genre_counts = {g: len(ids) for g, ids in genre_to_tracks.items()}
genre_counts_series = pd.Series(genre_counts).sort_values(ascending=False)
top_genres = genre_counts_series.head(top_n)
with st.expander("🎼 Genre distribution in your collection"):
    st.bar_chart(top_genres)
    st.write(f"Total tracks with genre: {sum(genre_counts.values())}")
    st.write(f"Unique genres: {len(genre_counts)}")
    st.write(f"Top genres: {', '.join(top_genres.index)}")

selected_genre = st.selectbox("Select genre for detailed metrics", list(top_genres.index))
tids = [tid for tid in genre_to_tracks[selected_genre] if tid in id_to_path]

def compute_metrics(recommended, relevant, similarities):
    metrics = {}
    for k in [5, 10, 20]:
        metrics[f'Precision@{k}'] = precision_at_k(recommended, relevant, k)
        metrics[f'Recall@{k}'] = recall_at_k(recommended, relevant, k)
        metrics[f'nDCG@{k}'] = ndcg(recommended, relevant, k)
    metrics['MRR'] = mean_reciprocal_rank(recommended, relevant)
    metrics['CosineMean@10'] = np.mean(similarities[:10])
    metrics['CosineStd@10'] = np.std(similarities[:10])
    metrics['CosineMean@20'] = np.mean(similarities[:20])
    metrics['CosineStd@20'] = np.std(similarities[:20])
    return metrics

uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

search_tab, metrics_tab = st.tabs(["🔍 Search", "📊 Metrics & Monitoring"])

with search_tab:
    if uploaded_file is not None:
        st.markdown(f"**Uploaded file:** `{uploaded_file.name}`")
        st.audio(uploaded_file, format="audio/mp3")
    if st.button("Find Similar Tracks"):
        if uploaded_file is not None:
            with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            st.session_state['uploaded_file_path'] = tmp_path
            try:
                start_time = time.time()
                results = model.find_similar_for_new_file(tmp_path, top_k=10)
                latency = time.time() - start_time
                st.success(f"Top 10 similar tracks for '{uploaded_file.name}': (Latency: {latency:.2f} sec)")
                for i, (track, dist) in enumerate(sorted(results, key=lambda x: x[1]), 1):
                    filename = os.path.basename(track)
                    st.write(f"{i}. {filename} (distance: {dist:.4f})")
                    try:
                        with open(track, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/mp3")
                    except Exception as e:
                        st.warning(f"Could not load audio: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please upload an MP3 file first.")

with metrics_tab:
    st.header(f"Offline Metrics for genre: {selected_genre}")
    try:
        if not tids:
            st.info("No tracks with embeddings for selected genre.")
        else:
            use_uploaded = 'uploaded_file_path' in st.session_state and os.path.exists(st.session_state['uploaded_file_path'])
            if use_uploaded:
                test_file_path = st.session_state['uploaded_file_path']
                test_id = os.path.splitext(os.path.basename(test_file_path))[0]
                relevant = [id_to_path[tid] for tid in tids if tid != test_id]
                results = model.find_similar_for_new_file(test_file_path, top_k=20)
                st.markdown(f"**Test file:** `{os.path.basename(test_file_path)}` (Uploaded by user, genre: `{selected_genre}`)")
                table_rows = []
                genre_query = track_to_genre.get(test_id, None)
                table_rows.append({
                    'Track': os.path.basename(test_file_path),
                    'Is Query': True,
                    'Genre': genre_query,
                    'Is Relevant': True,
                    'Similarity': 1.0
                })
                for track, sim in results:
                    tid = os.path.splitext(os.path.basename(track))[0]
                    genre = track_to_genre.get(tid, None)
                    is_relevant = (genre == genre_query)
                    table_rows.append({
                        'Track': os.path.basename(track),
                        'Is Query': False,
                        'Genre': genre,
                        'Is Relevant': is_relevant,
                        'Similarity': sim
                    })
                df_sim = pd.DataFrame(table_rows)
                st.subheader("Query & Top-N Similar Tracks Table")
                st.dataframe(df_sim)
                csv_sim = df_sim.to_csv(index=False).encode('utf-8')
                st.download_button("Download query+similar tracks table as CSV", csv_sim, f"similarity_table_{test_id}.csv", "text/csv")
            else:
                test_id = random.choice(tids)
                test_file_path = id_to_path[test_id]
                relevant = [id_to_path[tid] for tid in tids if tid != test_id]
                results = model.find_similar_for_file(test_file_path, top_k=20)
                st.markdown(f"**Test file:** `{os.path.basename(test_file_path)}` (Genre: `{selected_genre}`)")
            recommended = [os.path.basename(track) for track, sim in results]
            similarities = [sim for track, sim in results]
            metrics = compute_metrics(recommended, [os.path.basename(p) for p in relevant], similarities)
            st.markdown(f"**Relevant tracks (same genre, with embeddings):** {len(relevant)}")
            cols = st.columns(3)
            for idx, k in enumerate([5, 10, 20]):
                with cols[idx]:
                    st.metric(f"Precision@{k}", f"{metrics[f'Precision@{k}']:.3f}")
                    st.metric(f"Recall@{k}", f"{metrics[f'Recall@{k}']:.3f}")
                    st.metric(f"nDCG@{k}", f"{metrics[f'nDCG@{k}']:.3f}")
            st.metric("MRR", f"{metrics['MRR']:.3f}")
            # st.write(f"Cosine Similarity (top-10): mean={metrics['CosineMean@10']:.3f}, std={metrics['CosineStd@10']:.3f}")
            # st.write(f"Cosine Similarity (top-20): mean={metrics['CosineMean@20']:.3f}, std={metrics['CosineStd@20']:.3f}")
            # st.subheader("Similarity Distribution (top-20)")
            st.bar_chart(similarities)

            st.header(f"Aggregated metrics for genre '{selected_genre}' (up to 50 random tracks)")
            N = 50
            all_metrics = []
            track_ids = []
            for tid in random.sample(tids, min(N, len(tids))):
                relevant = [id_to_path[gtid] for gtid in tids if gtid != tid]
                if not relevant:
                    continue
                results = model.find_similar_for_file(id_to_path[tid], top_k=20)
                recommended = [os.path.basename(track) for track, sim in results]
                similarities = [sim for track, sim in results]
                metrics = compute_metrics(recommended, [os.path.basename(p) for p in relevant], similarities)
                all_metrics.append(metrics)
                track_ids.append(tid)
            if all_metrics:
                df_tracks = pd.DataFrame(all_metrics, index=track_ids).round(3)
                st.subheader(f"Detailed metrics for {len(df_tracks)} tracks of genre '{selected_genre}'")
                st.dataframe(df_tracks)
                csv_tracks = df_tracks.to_csv().encode('utf-8')
                st.download_button("Download track metrics as CSV", csv_tracks, f"metrics_{selected_genre}_tracks.csv", "text/csv")

                all_keys = set().union(*(m.keys() for m in all_metrics if isinstance(m, dict)))
                mean_metrics = {}
                for k in all_keys:
                    values = [m[k] for m in all_metrics if isinstance(m, dict) and k in m]
                    if values:
                        mean_metrics[k] = np.mean(values)
                if mean_metrics:
                    df = pd.DataFrame(mean_metrics, index=[selected_genre]).T.round(3)
                    #sort_metric = st.selectbox("Sort metrics table by", df.index, index=0)
                    #st.dataframe(df.sort_values(by=sort_metric, ascending=False))
                    col1, col2, col3 = st.columns(3)
                    for met, col in zip(['Precision@10', 'Recall@10', 'nDCG@10'], [col1, col2, col3]):
                        vals = [m[met] for m in all_metrics if isinstance(m, dict) and met in m]
                        if vals:
                            fig, ax = plt.subplots()
                            ax.hist(vals, bins=10, alpha=0.7)
                            ax.set_title(f'{met} distribution')
                            col.pyplot(fig)
                else:
                    st.info("Not enough valid metrics to aggregate.")
            else:
                st.info("Not enough tracks with ground truth for aggregated metrics.")

            st.header("Genre analytics (top-10 by count)")
            genre_table = {}
            genre_track_metrics = {}
            for genre in top_genres.index:
                tids_g = [tid for tid in genre_to_tracks[genre] if tid in id_to_path]
                if len(tids_g) < 5:
                    continue
                genre_metrics = []
                genre_track_ids = []
                for tid in random.sample(tids_g, min(10, len(tids_g))):
                    relevant = [id_to_path[gtid] for gtid in tids_g if gtid != tid]
                    if not relevant:
                        continue
                    results = model.find_similar_for_file(id_to_path[tid], top_k=20)
                    recommended = [os.path.basename(track) for track, sim in results]
                    similarities = [sim for track, sim in results]
                    metrics = compute_metrics(recommended, [os.path.basename(p) for p in relevant], similarities)
                    genre_metrics.append(metrics)
                    genre_track_ids.append(tid)
                if genre_metrics:
                    genre_table[genre] = {k: np.mean([m[k] for m in genre_metrics if k in m]) for k in genre_metrics[0]}
                    genre_track_metrics[genre] = pd.DataFrame(genre_metrics, index=genre_track_ids).round(3)
            if genre_table:
                df_genre = pd.DataFrame(genre_table).T.round(3)
                sort_metric2 = st.selectbox("Sort genre table by", df_genre.columns, index=0)
                st.dataframe(df_genre.sort_values(by=sort_metric2, ascending=False))
                csv_genre = df_genre.to_csv().encode('utf-8')
                st.download_button("Download genre metrics as CSV", csv_genre, "metrics_genres.csv", "text/csv")
                st.subheader("Detailed per-track metrics for each genre (top-10 genres)")
                for genre, df_g in genre_track_metrics.items():
                    with st.expander(f"{genre} ({len(df_g)} tracks)"):
                        st.dataframe(df_g)
                        csv_g = df_g.to_csv().encode('utf-8')
                        st.download_button(f"Download {genre} track metrics as CSV", csv_g, f"metrics_{genre}_tracks.csv", "text/csv")
    except Exception as e:
        st.warning(f"Could not calculate metrics: {e}")

def parse_genres_field(field):
    import ast
    try:
        return [int(x) for x in ast.literal_eval(field)]
    except Exception:
        return []

def build_multilabel_ground_truth(tracks, existing_ids):
    track_to_genres = {}
    genreid_to_title = {int(row['genre_id']): row['title'] for _, row in pd.read_csv('fma_metadata/genres.csv').iterrows()}
    for track_id in tracks.index:
        tid = str(track_id).zfill(6)
        if tid in existing_ids:
            genres_field = tracks.loc[track_id, ('track', 'genres')]
            genre_ids = parse_genres_field(genres_field) if pd.notnull(genres_field) else []
            genre_titles = [genreid_to_title.get(gid) for gid in genre_ids if gid in genreid_to_title]
            if genre_titles:
                track_to_genres[tid] = set(genre_titles)
    return track_to_genres

def jaccard_index(set1, set2):
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

track_to_genres_multi = build_multilabel_ground_truth(tracks, existing_ids)

with st.expander("🎵 Multi-label genre statistics"):
    genre_counts_per_track = [len(gs) for gs in track_to_genres_multi.values()]
    st.write(f"Mean genres per track: {np.mean(genre_counts_per_track):.2f}")
    st.bar_chart(np.bincount(genre_counts_per_track))

    N = 50
    valid_multi_ids = [tid for tid in track_to_genres_multi.keys() if tid in id_to_path]
    jaccards = []
    for tid in random.sample(valid_multi_ids, min(N, len(valid_multi_ids))):
        true_genres = track_to_genres_multi[tid]
        results = model.find_similar_for_file(id_to_path[tid], top_k=10)
        recommended_ids = [os.path.splitext(os.path.basename(track))[0] for track, sim in results]
        pred_genres = set()
        for rid in recommended_ids:
            pred_genres.update(track_to_genres_multi.get(rid, set()))
        jaccards.append(jaccard_index(true_genres, pred_genres))
    if jaccards:
        st.write(f"Mean Jaccard index (genres, top-10): {np.mean(jaccards):.3f}")
        fig, ax = plt.subplots()
        ax.hist(jaccards, bins=10, alpha=0.7)
        ax.set_title('Jaccard index (multi-label) distribution')
        st.pyplot(fig)

    genre_jaccard = {}
    genre_jaccard_std = {}
    genre_n_tracks = {}
    genre_mean_genres = {}
    genre_track_jaccards = {}
    for genre in genre_counts:
        tids = [tid for tid, gs in track_to_genres_multi.items() if genre in gs and tid in id_to_path]
        if len(tids) < 5:
            continue
        genre_j = []
        for tid in random.sample(tids, min(10, len(tids))):
            true_genres = track_to_genres_multi[tid]
            results = model.find_similar_for_file(id_to_path[tid], top_k=10)
            recommended_ids = [os.path.splitext(os.path.basename(track))[0] for track, sim in results]
            pred_genres = set()
            for rid in recommended_ids:
                pred_genres.update(track_to_genres_multi.get(rid, set()))
            genre_j.append(jaccard_index(true_genres, pred_genres))
        if genre_j:
            genre_jaccard[genre] = np.mean(genre_j)
            genre_jaccard_std[genre] = np.std(genre_j)
            genre_n_tracks[genre] = len(tids)
            genre_mean_genres[genre] = np.mean([len(track_to_genres_multi[tid]) for tid in tids])
            genre_track_jaccards[genre] = genre_j
    if genre_jaccard:
        df_jaccard = pd.DataFrame({
            'Mean Jaccard': genre_jaccard,
            'Std Jaccard': genre_jaccard_std,
            'N tracks': genre_n_tracks,
            'Mean genres per track': genre_mean_genres
        }).round(3)
        sort_col = st.selectbox("Sort multi-label genre table by", df_jaccard.columns, index=0)
        st.dataframe(df_jaccard.sort_values(by=sort_col, ascending=False))
        csv_jaccard = df_jaccard.to_csv().encode('utf-8')
        st.download_button("Download multi-label genre table as CSV", csv_jaccard, "multilabel_genre_jaccard.csv", "text/csv")

if 'df_jaccard' in locals():
    st.subheader("Detailed per-track Jaccard for each genre (top-10 genres)")
    for genre in list(df_jaccard.sort_values(by=sort_col, ascending=False).index)[:10]:
        tids = [tid for tid, gs in track_to_genres_multi.items() if genre in gs and tid in id_to_path]
        track_rows = []
        for tid in tids:
            true_genres = track_to_genres_multi[tid]
            results = model.find_similar_for_file(id_to_path[tid], top_k=10)
            recommended_ids = [os.path.splitext(os.path.basename(track))[0] for track, sim in results]
            pred_genres = set()
            for rid in recommended_ids:
                pred_genres.update(track_to_genres_multi.get(rid, set()))
            jac = jaccard_index(true_genres, pred_genres)
            track_rows.append({
                'Track ID': tid,
                'Jaccard': jac,
                'N genres': len(true_genres),
                'Genres': ', '.join(sorted(true_genres)),
                'Pred genres': ', '.join(sorted(pred_genres))
            })
        df_tracks = pd.DataFrame(track_rows).round(3)
        with st.expander(f"{genre} ({len(df_tracks)} tracks)"):
            st.dataframe(df_tracks)
            csv_t = df_tracks.to_csv(index=False).encode('utf-8')
            st.download_button(f"Download {genre} per-track Jaccard as CSV", csv_t, f"multilabel_{genre}_tracks.csv", "text/csv")
