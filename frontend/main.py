import sys
import os
import streamlit as st
from tempfile import NamedTemporaryFile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ML import model

st.title("Music Track Similarity Search")

uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

if st.button("Find Similar Tracks"):
    if uploaded_file is not None:
        with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            results = model.find_similar_for_new_file(tmp_path, top_k=10)
            st.success(f"Top 10 similar tracks for '{uploaded_file.name}':")
            for i, (track, dist) in enumerate(sorted(results, key=lambda x: x[1], reverse=True), 1):
                filename = os.path.basename(track)
                st.write(f"{i}. {filename} (similarity: {dist:.4f})")
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
