import nltk

from nltk.book import *
from __future__ import division

#show occurence of a given word with surrounding contexts
text1.concordance("monstrous")

#show words appeared in a similar context
text1.similar("monstrous")

#dispersion plot
text4.dispersion_plot(["citizen","democracy","freedom"])

#length of tokens
len(text3)

#vocabulary
set(text3)

#lexical richness
#each word is used 7.42 times on average
len(text5)/len(set(text5))

#word-frequency distribution
frequencyDist1 = FreqDist(text1)
#wordFreqs in {"word":<freq>,...}
wordFreqs = frequencyDist1
#generate cummulative plot
frequencyDist1.plot(50,cumulative=True)
#hapaxes -> words that occur only once
frequencyDist1.hapaxes()

#collocation -> sequence of words that occur together unsually often. Resistance to substitutions.
#create bigram
from nltk.util import bigrams
list(bigrams(['hello','world','this','is','a','nice','day']))
#retrieve collocations
text4.collocations()
