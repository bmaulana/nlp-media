from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = CountVectorizer()
corpus = [
    'This is the first document.',
    'This is the second document document.',
    'And the third one.',
    'Is this the first document?',
    "Does it split words like John's or They've?"
]

matrix = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(vectorizer.vocabulary_)
# it DOES lowercase input and split on ',
# but doesn't recognise "'s" as a separate topic, but "ve" is (>1 character)
print(matrix.toarray())

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(matrix)
print(tfidf.toarray())

keyword_index = tfidf[:, vectorizer.vocabulary_.get('document')].toarray().reshape([1, 5])
print(type(keyword_index))
keyword_index.sort()
print(keyword_index)
print(sum(keyword_index) / len(keyword_index))
