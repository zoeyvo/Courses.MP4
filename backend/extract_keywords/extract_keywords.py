import string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def extract_keywords(desc : str):
    # Extracts keywords from the passed course description
    # @param desc: the course description
    # @returns a list of keywords

    # referenced from Maarten Grootendorst's guide on keyword extraction
    n_gram_range = (1, 2)
    stop_words = "english"

    count = CountVectorizer(ngram_range = n_gram_range, stop_words=stop_words).fit([desc])
    candidates = count.get_feature_names_out()

    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    doc_embedding = model.encode([desc])
    candidate_embeddings = model.encode(candidates)

    top_n = 10
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    return keywords

