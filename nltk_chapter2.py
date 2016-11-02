#Managing Text Corpora object
#Loading new corpus from files
#Conditional distribution table
#Lexical Resources - consisted of lemma(headword), sense(meaning), POS
#WordList, StopWordList Corpora
#WordNet Corpora -> Synonym, Hierarchy, Containment, Entailment, Antonymy

import nltk
from nltk.book import *
from __future__ import division

#Project Gutenberg -> contains 25,000 free electronic books
#.fileids() -> list file's id
nltk.corpus.gutenberg.fileids()
#load file as a Words
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
#load file as a Sentences
emma = nltk.corpus.gutenberg.sents('austen-emma.txt')
#look at text
" ".join(emma[0:100])
#lexical richness
len(emma)/len(set(emma))
#get raw text
from nltk.corpus import gutenberg
gutenberg.raw('austen-emma.txt')
#lexical diversity score
#(avg word length, avg sentence length, lexical richness)
#can used to investigate characteristics of author
for fileId in gutenberg.fileids():
    charCount = len(gutenberg.raw(fileId))
    wordCount = len(gutenberg.words(fileId))
    sentenceCount = len(gutenberg.sents(fileId))
    vocabSize = len(set([w.lower() for w in gutenberg.words(fileId)]))
    print("{} {} {} {}".format(int(charCount/wordCount),int(wordCount/sentenceCount),int(wordCount/vocabSize),fileId))
    
#Web Chat
from nltk.corpus import nps_chat
#contains chat in each room, divided by age group
#ex: 10-19-20s_706posts.xml -> rooms of age~20, date 19 OCT, contains 706 posts(messages)
nps_chat.fileids()
chatroom = nps_chat.posts("10-19-20s_706posts.xml")
#list contains instance of each send message
chatroom[0:10]
#posts is the number of messages
len(chatroom)

#**Brown Corpus
#contains > million words
#contains > 500 sources, organized by genre(news,editorial,...)
#good for investigate different among genres
from nltk.corpus import brown
#show categories(genres)
brown.categories()
#select only fileIds in news category
brown.fileids(categories="news")
#select all words in category news
brown.words(categories="news")
#*the book suggests frequency distribution of modal verbs:['can','could','may','might','must','will'] exposes different among genres. Romance genre tends to use 'could' while News genre tends to use 'will'.

#Reuters Corpus: news topic, classified into 90 topics
from nltk.corpus import reuters
#data split into train & test set
reuters.fileids()
reuters.categories()

#Inaugural Address
#fileId come with a year
from nltk.corpus import inaugural
inaugural.fileids()

#Other important corpus
#*Movie Reviews -> 2k movie reviews with sentiment polarity classification
#* Penn Treebank -> 40k words, tagged and parsed
#* Switchboard -> 36 phone calls
#* WordNet -> 145k synonym sets

#basic corpus's method, can go to http://www.nltk.org/howto
help(nltk.corpus.reader)
#frequent used functions:
#{corpus} .fileids(), .raw(), .words(), .sents(), .categories()

#Load your own corpus
from nltk.corpus import PlaintextCorpusReader
#corpus contains all files in /tmp
corpus = PlaintextCorpusReader("/tmp",".*")
corpus.fileids()

#Conditional Frequency Distribution
#Counting words by group
genreWords = [("A","a"),("A","b"),("B","b")]
cfd = nltk.ConditionalFreqDist(genreWords)
#get groups
cfd.conditions()
#count for group A
cfd['A']
#tabulate counting table
#use parameter: conditions and samples to explore only some parts of the table
cfd.tabulate(conditions=['A','B'],samples=['a','b'],cumulative=True)

#Lexical Resources
#Homonyms -> words with the same spelling but have different sense

#Wordlist Corpora -> used for spell checker
wordlists = nltk.corpus.words.words()

#StopWord Corpora
stopwordLists = nltk.corpus.stopwords.words('english')

#WordNet Corpora -> contains word synonym sets
from nltk.corpus import wordnet as wordnet
#find synonym set of 'motorcar'
#[Synset('car.n.01')]
#-> has one synset(synonym set) (mean has only one possible meaning(sense))
synsets = wordnet.synsets('motorcar')
#words in that synset is called lemmas
synset = synsets[0]
#print lemmas in synset
synset.lemma_names()
#print definition
synset.definition()
#print example usage
synset.examples()
#to access all lemmas of the word 'car'
wordnet.lemmas('car')
#to access a particular synset
wordnet.synset('car.n.02').definition()

#WordNet Hierarchy(Is-A relationship)
wordnet.synset('car.n.01').definition()
wordnet.synset('car.n.01').lemma_names()
#Look at Hyponyms -> specific instances
#return a list of Hyponyms Synsets
wordnet.synset('car.n.01').hyponyms()
wordnet.synset('car.n.01').hyponyms()[0].definition()
wordnet.synset('car.n.01').hyponyms()[0].lemma_names()
#Look at Hypernyms -> more general instances
wordnet.synset('car.n.01').hypernyms()
#Get root Hypernyms
wordnet.synset('car.n.01').root_hypernyms()

#WordNet Component relationship
#meronyms -> the item A
#holonyms -> items that contain item A
#ex: find sub components of synset 'tree'
wordnet.synset('tree.n.01').part_meronyms()
#find 'substance' of tree
wordnet.synset('tree.n.01').substance_meronyms()
#find holonyms(trees constitute a forest)
wordnet.synset('tree.n.01').member_holonyms()

#WordNet Entailment relationship
wordnet.synset('walk.v.01').entailments()

#WordNet Lemms Antonym relationship
wordnet.lemma('supply.n.02.supply').antonyms()

#Hierachy can be useful to compare similarity between synsets
#'baleen whale' is more specific that 'whale'
wordnet.synset('baleen_whale.n.01').min_depth()
wordnet.synset('whale.n.01').min_depth()
wordnet.synset('panda.n.01').min_depth()
#path_similarity -> shortest path that linked two synset
#'baleen whale' is more similar to 'panda' that 'car'
wordnet.synset('baleen_whale.n.01').path_similarity(wordnet.synset('panda.n.01'))
wordnet.synset('baleen_whale.n.01').path_similarity(wordnet.synset('car.n.01'))
#There are more similarity measurement that come with WordNet
