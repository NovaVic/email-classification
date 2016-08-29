from nltk.classify import NaiveBayesClassifier


def train_classifier(label_feat):
    if not hasattr(train_classifier, "nb_classifier"):
        train_classifier.nb_classifier =  NaiveBayesClassifier.train(label_feat)
    return train_classifier.nb_classifier 


def classify_input(input_feat, nb_classifier):
    return nb_classifier.classify(input_feat)
