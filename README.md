# Music Similarity Search (FAISS + YAMNet, Streamlit)

Search for similar music tracks using YAMNet embeddings and fast local FAISS search.  
The Streamlit web interface allows you to upload tracks, search for similar ones, and listen to results directly in your browser.

---

## Features

- **AI-based music analysis:** Extract embeddings using YAMNet (TensorFlow Hub)
- **Local search:** Fast search for similar tracks using FAISS (no internet, no cloud)
- **Web interface:** Convenient Streamlit UI with the ability to listen to found tracks
- **Scalability:** Supports large local collections (tens and hundreds of thousands of mp3s)
- **Simplicity:** No cloud services, just local files and Python

---

## Requirements

- Python 3.11+
- `streamlit==1.28.1`
- `numpy>=1.26.0,<2.2.0`
- `tqdm==4.66.1`
- `faiss-cpu==1.7.4`
- `librosa==0.10.1`
- `python-dotenv==1.0.0`
- `soundfile==0.12.1`
- `tensorflow==2.19.0`
- `tensorflow-hub`
- MP3 files (FMA structure recommended: `fma_large/000/000001.mp3`)

Download the dataset (fma_large!) from here:
https://github.com/mdeff/fma?tab=readme-ov-file

To create embeddings:
```bash
git clone URL-this-project.git
```
```bash
cd this-project
```
```bash
python -m venv .venv
```
For Linux/Mac:
```bash
source .venv/bin/activate
```
For Windows:
```bash
.venv/Scripts/activate
```
```bash
pip install -r requirements.txt
```
```bash
cd ML
```
```bash
python model.py
```

To start the project:
```bash
streamlit run frontend/main.py
```

---

## Project Structure

```
dls-pr/
├── frontend/
│   └── main.py           # Streamlit interface for search and playback
├── ML/
│   ├── model.py          # FAISS + YAMNet search logic
│   └── test_search.py    # Examples and tests
├── fma_large/            # Your mp3 collection (FMA structure)
├── requirements.txt
└── README.md
```

---

## Quick Start

1. **Prepare your mp3 collection**  
   Place all your mp3 files in the `fma_large/` folder (nested structure supported).

2. **Extract embeddings**  
   On first run, the search will automatically create embeddings for all mp3s (may take some time).

3. **Run the web interface**  
   ```bash
   streamlit run frontend/main.py
   ```
   Open your browser: [http://localhost:8501](http://localhost:8501)

4. **Enjoy!**
   - Upload an mp3 file via the interface
   - Get the top-10 similar tracks with similarity score (the higher, the better)
   - Listen to results directly in your browser

---

## How it works

1. **Feature extraction:**  
   Each mp3 file is processed by YAMNet (tensorflow_hub), resulting in a 1024-dimensional embedding.

2. **Index building:**  
   All embeddings are indexed using FAISS (cosine similarity search).

3. **Search:**  
   The uploaded track is also converted to an embedding, and the most similar tracks are found in the index.

4. **Results:**  
   A list of similar tracks is displayed with their names, similarity (the higher, the better), and an audio player for each result.

---

## Example Output

```
1. 070085.mp3 (similarity: 0.9239)
[audio player]
2. 044864.mp3 (similarity: 0.9213)
[audio player]
...
```

---

## Usage in Code

```python
from ML import model

results = model.find_similar_for_new_file("path/to/song.mp3", top_k=10)
for path, similarity in results:
    print(path, similarity)
```

---

## Metrics & Analytics

The web interface provides detailed analytics for evaluating the quality of music recommendations:

- **Offline metrics:**
  - Precision@K, Recall@K, nDCG@K, MRR, Cosine Similarity
  - Calculated for each genre and for uploaded tracks
- **Genre analytics:**
  - Aggregated metrics for top genres
  - Per-track metrics tables for each genre
  - Downloadable CSV tables for further analysis
- **Multi-label genre statistics:**
  - Jaccard index for genre overlap between recommendations and ground truth
  - Distribution of genres per track
  - Detailed per-track Jaccard tables for each genre

All statistics and tables are available in the Streamlit interface after uploading your collection and/or a test track.

---

## License

MIT
