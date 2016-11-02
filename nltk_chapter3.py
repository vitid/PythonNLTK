# Processing Raw Text(Tokenization & Stemming)

import nltk
from __future__ import division
import re, pprint

# Tokenization
tokens = nltk.word_tokenize("hello world this is a nice day, hello world again")
# this is just a list
type(tokens)
tokens
# to make it be able to processed by techniques in Chapter I, put the tokenized tokens into nltk.Text()
corpus = nltk.Text(tokens)
type(corpus)
corpus[0]
tokens.collocations()
tokens.concordance("is")
# Customized Tokenization with RegEx
# Noted!! -> (...) must be non-capture parenthesis(prefix with ?: such as (?:...)
text = 'That U.S.A. poster-print costs $12.40... , .'
pattern = r'''(?x)  # set flag to allow comments
         (?:[A-Z]\.)+       # abbreviations, e.g. U.S.A.
         |\.\.\.            # capture ...
         |\w+(?:-\w+)*      #word with optional hyphens
         |\$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
         | [][.,;"'?():-_`] # these are separate tokens
'''
regexTokenizer = nltk.tokenize.RegexpTokenizer(pattern)
regexTokenizer.tokenize(text)
# Off-the-self Tokenizer -> split some punctuations and also keep them
# May not be good enough...
nltk.word_tokenize("hello ; hello; hello. . this.is this's")

# load gutenberg corpus
from nltk.corpus import gutenberg
gutenberg.fileids()
corpus = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
corpus.collocations()
# regular expression
# (...) returned only words in (...)
corpus.findall(r"<a> (<.*>) <man>")
corpus.findall(r"<a> (<.*>) <woman>")
corpus.findall(r"<.*> <.*> <man>")

# Text Normalization
raw = """DENNIS: Listen, strange women lying in ponds distributing swords
is no basis for a system of government. Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""
tokens = nltk.word_tokenize(raw)
tokens
# Porter Stemmers -> good for word indexing, information retrieval
porter = nltk.PorterStemmer()
stems = [porter.stem(t) for t in tokens]
zip(tokens,stems)
# WordNet Lemmatization -> good for pairing with WordNet relationship lookup
wordNetStemmer = nltk.WordNetLemmatizer()
stems = [wordNetStemmer.lemmatize(t) for t in tokens]
zip(tokens,stems)