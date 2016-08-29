import collections

 
def bag_of_words(words):
   return dict([(word, True) for word in words])


def  load_label_features_from_corpus(subject_body, label, labels):
     #adding attribute to function to use it like a static var

     if not hasattr(load_label_features_from_corpus, "label_feats"):
         load_label_features_from_corpus.label_feats = [] 
         print(load_label_features_from_corpus.label_feats)
  
     feats =  bag_of_words(subject_body)
     load_label_features_from_corpus.label_feats += [(feats, label)]
     #print( load_label_features_from_corpus.label_feats)
     return  load_label_features_from_corpus.label_feats
