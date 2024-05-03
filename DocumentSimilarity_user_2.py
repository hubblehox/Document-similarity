import numpy as np
import openai
from flask import Flask, request
app = Flask(__name__)

class DocumentSimilarity:
    def __init__(self, documents, model="text-embedding-ada-002", threshold=0.7):
        self.documents = documents
        self.model = model
        self.threshold = threshold
        self.vectors = self._embed_documents()
        self.similar_documents = self._calculate_similarities()

    def _embed_documents(self):
        vectors = []
        for doc in self.documents:
            response = openai.Embedding.create(input=[doc], model=self.model)
            vector = response["data"][0]["embedding"]
            vectors.append(vector)
        return vectors

    def _calculate_similarities(self):
        similar_documents = []
        for i in range(len(self.vectors)):
            for j in range(i+1, len(self.vectors)):
                cosine_similarity = np.dot(self.vectors[i], self.vectors[j]) / (
                    np.linalg.norm(self.vectors[i]) * np.linalg.norm(self.vectors[j])
                )
                if cosine_similarity > self.threshold:
                    similar_documents.append((self.documents[i], self.documents[j], cosine_similarity))
        similar_documents.sort(key=lambda x: x[2], reverse=True)
        return similar_documents

    def get_similar_documents(self, top_n):
        similar_docs = []
        for doc in self.similar_documents[:top_n]:
            similar_docs.append({
                'document1': doc[0],
                'document2': doc[1],
                'similarity': doc[2]
            })
        return similar_docs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the file paths from the user
        text1 = request.form.get('text1')
        text2 = request.form.get('text2')

        # Create a list of the document paths
        docs = [text1, text2]

        # Create an instance of the DocumentSimilarity class
        ds = DocumentSimilarity(docs)

        # Get the top 3 similar documents
        similar_docs = ds.get_similar_documents(2)

        # Format and return the response
        response = ""
        for doc in similar_docs:
            response += f"{doc['document1']} is similar to {doc['document2']} with a cosine similarity of {doc['similarity']}<br>"
        return response

    # Render the HTML form for file input
    return '''
        <form method="POST">
            <label for="text1">Text 1:</label><br>
            <input type="text" id="text1" name="text1"><br><br>
            <label for="text2">Text 2:</label><br>
            <input type="text" id="text2" name="text2"><br><br>
            <input type="submit" value="Submit">
        </form>
    '''

if __name__ == '__main__':
    openai.api_key = ""
    app.run()
