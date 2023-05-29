from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocumentRetriever:

    def __init__(self, model_name='facebook/dpr-ctx_encoder-single-nq-base'):
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
        self.context_model = DPRContextEncoder.from_pretrained(model_name)
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        self.question_model = DPRQuestionEncoder.from_pretrained(model_name)

    def embed_text(self, text, model, tokenizer):
        # Tokenization
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        # Embedding
        outputs = model(**inputs)
        embeddings = outputs.pooler_output
        return embeddings

    def retrieve(self, documents, query):
        # Embed documents
        document_embeddings = []
        for doc in documents:
            doc_emb = self.embed_text(doc, self.context_model, self.context_tokenizer)
            document_embeddings.append(doc_emb.detach().numpy())

        # Embed the query
        query_embedding = self.embed_text(query, self.question_model, self.question_tokenizer)
        query_embedding = query_embedding.detach().numpy()

        # Compare embeddings to find the most relevant document
        similarities = [cosine_similarity(query_embedding, doc_emb)[0][0] for doc_emb in document_embeddings]
        most_relevant_idx = np.argmax(similarities)

        return documents[most_relevant_idx], similarities[most_relevant_idx]


if __name__ == "__main__":
    documents = ["This is a sample document 1.", 
                 "This is another sample document 2.", 
                 "This is yet another sample document 3."]

    query = "What is in document 1?"

    dr = DocumentRetriever()

    relevant_doc, similarity = dr.retrieve(documents, query)

    print(f"The most relevant document is: {relevant_doc} with a similarity of {similarity}")
