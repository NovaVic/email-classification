# writing an email classifier using various python nltk libraries
# working on this https://community.topcoder.com/longcontest/?module=ViewProblemStatement&compid=53001&rd=16771
# during my free time

import nltk
from nltk.corpus.reader import TaggedCorpusReader
from nltk.tokenize import LineTokenizer
from bs4 import BeautifulSoup

from  feat_extract import load_label_features_from_corpus
from feat_extract import bag_of_words
from classification  import train_classifier
from classification import classify_input

# example:
# fileName '/home/suraiya/nltk_data/topcoder_train'
# dirName = 'email-classification-training.csv'

fileName = input("Enter your filename with training dataset: ");
dirName = input("Enter your directory name for your file: ");

reader =  TaggedCorpusReader(dirName, fileName, sent_tokenizer=LineTokenizer())

from nltk.corpus import stopwords
from nltk.tokenize import  WordPunctTokenizer
from nltk.tokenize import RegexpTokenizer
import re

SUBJECT = 1
BODY = 2
CATEGORY = 3
TRAINING_COUNT = 1500
TEST_BATCH_SIZE = 500
STOP_AT_BATCH = TRAINING_COUNT + TEST_BATCH_SIZE

#do something based on language detection
stop_words = set(stopwords.words('english'))

#adding pre and ref to the list of regular stop words
stop_words.update(['pre', 'ref'])
pat = re.compile(r'\[|\]|\'|\(|\)|,|>|<|#|"')

#print(stop_words)

batch_count=0

from nltk.corpus import wordnet

#use the symbols as separator or gap symbols
tokenizer = RegexpTokenizer(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", gaps=True);

tokenizer_wpt = WordPunctTokenizer()
classifier = None
for r in reader.tagged_sents():
  #removing tags for all  tags is equivalent to
  #soup.get_text()
  #if we remove few tags and leave others intact then  calling custom remove_tags function makes sense
 
  soup =   BeautifulSoup(str(r)).get_text() 
  
  important_content = ""

  field_count  = 0

  subject = ""
  body = ""
  category = ""
  categories =  set()

  for field in tokenizer.tokenize(soup):
     field_merged_back =   list(); 
     words = tokenizer_wpt.tokenize(pat.sub('', field))

     for word in words:
         word = word.lower() 
         if word != 'none' and word.strip() != '' and word not in stop_words:
            field_merged_back.append( word )
         
     if field_count  == SUBJECT:
         subject =  field_merged_back
         
     elif field_count == BODY:
            print(field_merged_back)
            body =  field_merged_back

     elif field_count == CATEGORY:
            category =  " ".join(field_merged_back)
            categories.update([category])
                     
     field_count += 1
                
  label_feats = load_label_features_from_corpus( body, category, categories)
  
  batch_count += 1
  if batch_count == TRAINING_COUNT:
     classifier = train_classifier(label_feats)

  elif batch_count >  TRAINING_COUNT:
     #use the data for testing
     print(category) #input category
     print(classify_input(bag_of_words(body), classifier)) #Category emitted by classifier

  if batch_count == STOP_AT_BATCH:
     break 

