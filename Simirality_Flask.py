from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/cosine_similarity', methods=['POST'])
def calculate_cosine_similarity():
    if 'file1' not in request.files or 'file2' not in request.files:
        return "No file part"
    file1 = request.files['file1']
    file2 = request.files['file2']

    # Convert files to text
    text1 = file1.stream.read().decode("utf-8") 
    text2 = file2.stream.read().decode("utf-8")
    print(text1)
    texts = [text1, text2]

    # Get embeddings
    embeddings = model.encode(texts)

    # Calculate cosine similarity
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])

    # Convert numpy float32 to Python float before returning
    return {'cosine_similarity': float(cos_sim[0][0])}

if __name__ == '__main__':
    app.run(debug=True)
