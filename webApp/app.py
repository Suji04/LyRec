import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from flask import Flask, request, render_template
import pandas as pd
from annoy import AnnoyIndex

print("Loading the model...")
embedding_model = SentenceTransformer(
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    device='mps',
    model_kwargs={"torch_dtype": torch.float16}
)
embedding_model.max_seq_length = 8192
print("Model is now loaded.")

print("Loading the database...")
root_dir = "database"
songs_df = pd.read_csv(f"{root_dir}/60k_songs.csv")

lyrics_embeddings = np.load(f"{root_dir}/60k_song_lyrics_embeddings.npy")
summary_embeddings = np.load(f"{root_dir}/60k_song_summary_embeddings.npy")

embedding_dim_lyrics = lyrics_embeddings.shape[1]
embedding_dim_summary = summary_embeddings.shape[1]

lyrics_index = AnnoyIndex(embedding_dim_lyrics, metric='dot')
for i, v in enumerate(lyrics_embeddings):
    lyrics_index.add_item(i, v)
lyrics_index.build(25)  

summary_index = AnnoyIndex(embedding_dim_summary, metric='dot')
for i, v in enumerate(summary_embeddings):
    summary_index.add_item(i, v)
summary_index.build(25)

print("Annoy indexes are built.")
print("Database is now loaded.")

class LyRec:
    def __init__(self, songs_df, lyrics_index, summary_index, embedding_model):
        self.songs_df = songs_df
        self.lyrics_index = lyrics_index
        self.summary_index = summary_index
        self.embedding_model = embedding_model

    def get_records_from_id(self, song_ids):
        songs = []
        for _id in song_ids:
            songs.extend(
                self.songs_df[self.songs_df["song_id"] == (_id+1)].to_dict(orient='records')
            )
        return songs

    def get_songs_with_similar_lyrics(self, query_lyrics, k=10):
        prompt = f"Instruct: Given the lyrics, retrieve relevant songs\n Query: {query_lyrics}"
        query_embedding = self.embedding_model.encode(prompt).astype(np.float32)
        song_ids = self.lyrics_index.get_nns_by_vector(query_embedding, k, include_distances=False)
        return self.get_records_from_id(song_ids)

    def get_songs_with_similar_description(self, query_description, k=10):
        prompt = f"Instruct: Given a description, retrieve relevant songs\n Query: {query_description}"
        query_embedding = self.embedding_model.encode(prompt).astype(np.float32)
        song_ids = self.summary_index.get_nns_by_vector(query_embedding, k, include_distances=False)
        return self.get_records_from_id(song_ids)

    def get_songs_with_similar_lyrics_and_description(self, query_lyrics, query_description, k=10):
        prompt_lyrics = f"Instruct: Given the lyrics, retrieve relevant songs\n Query: {query_lyrics}"
        query_lyrics_embedding = self.embedding_model.encode(prompt_lyrics).astype(np.float32)
        top_n = 500
        top_lyrics_ids = self.lyrics_index.get_nns_by_vector(query_lyrics_embedding, top_n, include_distances=False)
        

        embedding_dim = self.summary_index.f
        temp_index = AnnoyIndex(embedding_dim, metric='dot')
        
        for i, song_id in enumerate(top_lyrics_ids):
            temp_index.add_item(i, summary_embeddings[song_id])
        temp_index.build(25)
        
        prompt_desc = f"Instruct: Given a description, retrieve relevant songs\n Query: {query_description}"
        query_desc_embedding = self.embedding_model.encode(prompt_desc).astype(np.float32)
        
        final_local_ids = temp_index.get_nns_by_vector(query_desc_embedding, k, include_distances=False)
        final_song_ids = [top_lyrics_ids[i] for i in final_local_ids]

        return self.get_records_from_id(final_song_ids)


lyrec = LyRec(songs_df, lyrics_index, summary_index, embedding_model)




app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')


import random

@app.route('/search', methods=['POST'])
def search():
    query_lyrics = request.form.get('query_lyrics', '').strip()
    query_description = request.form.get('query_description', '').strip()

    if query_lyrics and query_description:
        results = lyrec.get_songs_with_similar_lyrics_and_description(query_lyrics, query_description, k=5)
    elif query_lyrics:
        results = lyrec.get_songs_with_similar_lyrics(query_lyrics, k=5)
    elif query_description:
        results = lyrec.get_songs_with_similar_description(query_description, k=5)
    else:
        results = []

    # Assign a random background color to each song container
    color_choices = [
        "#033F63", "#28666E", "#7C9885", "#B5B682", 
        "#CE4257", "#FF7F51", "#FF9B54",
    ]
    colors = random.sample(color_choices, 5)
    for song, c in zip(results, colors):
        song["_bg_color"] = c
    
    return render_template('results.html', songs=results)


if __name__ == '__main__':
    app.run(debug=True)
