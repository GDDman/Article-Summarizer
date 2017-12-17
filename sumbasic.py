import sys
import os
import codecs
import random
import math

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

stop_list = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def wordnet_pos(pos):

    if pos.startswith('J'):
        return wordnet.ADJ
    elif pos.startswith('V'):
        return wordnet.VERB
    elif pos.startswith('N'):
        return wordnet.NOUN
    elif pos.startswith('R'):
        return wordnet.ADV
    else:
        return None

def ispunct(string):
    return not any(char.isalnum() for char in string)

def preprocess(sentences):

    # lowercase, punctuation, lemmatize, stopwords
    processed = {}
    for i, line in enumerate(sentences):
        newline = line.lower()
        newline = word_tokenize(newline.strip())
        newline = pos_tag(newline)
        for j, (w, pos) in enumerate(newline):
            pos = wordnet_pos(pos)
            if pos:
                newline[j] = lemmatizer.lemmatize(w, pos)
            else:
                newline[j] = lemmatizer.lemmatize(w)
        newline = [w for w in newline if w not in stop_list and not ispunct(w)]
        processed[i] = [line, ' '.join(newline)]
    return processed

def merge(files):

    merged = {}
    for i, f in enumerate(files):
        for j, sentence in f.items():
            merged[str(i) + '-' + str(j)] = sentence
    return merged

def word_probs(sentences):

    # get probabilities of words appearing
    wordprobs = {}
    total = 0.
    for sentence in sentences.values():
        words = sentence[1].split()
        for w in words:
            total += 1.
            if w in wordprobs:
                wordprobs[w] += 1.
            else:
                wordprobs[w] = 1.
    for w in wordprobs.keys():
        wordprobs[w] /= total
    return wordprobs

def get_sentence_scores(sentences, wordprobs):

    # score a sentence
    sentence_scores = {}
    for i, sentence in sentences.items():
        words = sentence[1].split()
        if len(words) is 0:
            score = 0
        else:
            score = math.fsum([wordprobs[w] for w in words]) / len(words)
        if score in sentence_scores:
            sentence_scores[score].append(i)
        else:
            sentence_scores[score] = [i]
    return sentence_scores

def sumbasic(files, orig, limit):

    sentences = merge(files)
    wordprobs = word_probs(sentences)
    scores = get_sentence_scores(sentences, wordprobs)

    summary = ""
    length = 0
    while length < limit:
        ids = scores[max(scores.keys())]
        i = ids[random.randint(0, len(ids)-1)]
        summary += sentences[i][0] + ' '
        length += len(sentences[i][0].split())
        if orig:
            # update word probabilities
            for w in sentences[i][1].split():
                wordprobs[w] *= wordprobs[w]
            scores = get_sentence_scores(sentences, wordprobs)
        else:
            # remove sentence
            scores.pop(max(scores.keys()))

    print(summary)
    return summary


def leading(files, limit):

    # string together first sentences of random files
    summary = ""
    length = 0
    index = random.randint(0, len(files)-1)
    for i, sentence in files[index].items():
        if length < limit:
            summary += sentence[0]+' '
            length += len(sentence[0].split())
        else:
            break
    print(summary)
    return summary

def main():

    # length of summary
    limit = 100

    sumtype = sys.argv[1]
    fname = sys.argv[2]
    split = fname.rsplit('/', 1)
    path = split[0] + '/'
    ext = split[1].rsplit('-', 1)[0]

    filenames = []

    # get files that match
    allfiles = os.listdir(path)
    for name in allfiles:
        if name.startswith(ext):
            filenames.append(name)

    # change filenames into files with sentence tokens
    files = []
    for name in filenames:
        sentences = codecs.open(path + name, 'r', encoding='utf-8').read()
        sentences = codecs.encode(sentences, 'ascii', 'ignore')
        sentences = sent_tokenize(sentences.decode())
        files.append(preprocess(sentences))

    if sumtype == 'orig':
        sumbasic(files, True, limit)
    elif sumtype == 'simplified':
        sumbasic(files, False, limit)
    elif sumtype == 'leading':
        leading(files, limit)
    else:
        print('method must be orig, simplified, or leading.')


if __name__ == '__main__':
    main()
