import csv
import random
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

# POS_TAGGER_FUNCTION : TYPE 1
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None

#load definitions
glosses_data = "./data/definitions/wkt/transformer_wkt_DSR_model.csv"#
out_file = open("wkt_training_triples.txt", "w")

#process glosses data
with open(glosses_data, newline='', encoding='ISO-8859-1') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
    for row in spamreader:
        print("converting...", row[0], row[1], row[2])
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(row[3])) 
        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
        for definendum in row[2].split(", "):
            index = 0
            for definiens in row[4:]:
                definien_tag = definiens.split("/")
                if definien_tag[-1] =="O":
                    index += 1
                    continue
                if not definien_tag[0].lower() in stopwords.words("english") and definien_tag[0].lower().replace("-","").isalpha():
                    if wordnet_tagged[index][1] is None:
                    # if there is no available tag, append the token as is
                        print(definendum.lower()+"\t"+definien_tag[-1]+"\t"+wordnet_tagged[index][0].lower().replace("-","_"), file = out_file)
                    else:       
                    # else use the tag to lemmatize the token
                        print(definendum.lower()+"\t"+definien_tag[-1]+"\t"+lemmatizer.lemmatize(wordnet_tagged[index][0], wordnet_tagged[index][1]).lower().replace("-","_"), file = out_file)
                index += 1

out_file.close()