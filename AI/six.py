import collections
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.corpus.reader import CategorizedPlaintextCorpusReader

# Указываем путь к корпусу
corpus_root = "C:/Users/admin/PycharmProjects/kursov/feedback_parse"
reader = CategorizedPlaintextCorpusReader(corpus_root, r'.*\.txt', cat_pattern=r'(neg|pos)_(\w+)\.txt')

#категориальный корпус текстов
print("Категории:")
print(reader.categories())
print("Отрицательные:")
print(reader.fileids(categories=['neg']))
print("Положительные:")
print(reader.fileids(categories=['pos']))

#_____________________________________________
# стоп-слова на русском
stopWordsRu = set(stopwords.words('russian'))
stopWords = sorted(list(stopWordsRu))
print(stopWords)

# Создание словаря признаков
def bag_of_words(words):
    return {word: True for word in words}

def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))

def bag_of_non_stopwords(words, stopfile='russian'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)

# Создание коллекции списков признаков
def label_feats_from_corpus(corp, feature_decorator=bag_of_words):
    label_feats = collections.defaultdict(list)
    for label in corp.categories():
        for fileid in corp.fileids(categories=[label]):
            feats = feature_decorator(corp.words(fileids=[fileid]))
            label_feats[label].append(feats)
    return label_feats

# Разделение выборки отзывов на обучающую и тестовую
def split_label_feats(lfeats, split=0.75):
    train_feats = []
    test_feats = []
    for label, feats in lfeats.items():
        cutoff = int(len(feats) * split)
        train_feats.extend([(feat, label) for feat in feats[:cutoff]])
        test_feats.extend([(feat, label) for feat in feats[cutoff:]])
    return train_feats, test_feats

# Признаки корпуса
lfeats = label_feats_from_corpus(reader)
print("Классы в обучающих данных:", lfeats.keys())


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Обучение с стоп
train_feats, test_feats = split_label_feats(lfeats, split=0.75)

print("Количество обучающих отзывов с использованием стоп-слов:", len(train_feats))
print("Количество тестовых отзывов с использованием стоп-слов:", len(test_feats))

# Обучение классификатора
nb_classifier_with_stopwords = NaiveBayesClassifier.train(train_feats)

# Оценка качества
acc_val_with_stopwords = accuracy(nb_classifier_with_stopwords, test_feats)
print(f"Точность классификатора с использованием стоп-слов: {acc_val_with_stopwords:.2f}")

print("Слова с наибольшим влиянием на классификацию (с использованием стоп-слов):")
nb_classifier_with_stopwords.show_most_informative_features(10)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Обучение без стоп
lfeats_without_stopwords = label_feats_from_corpus(reader, feature_decorator=bag_of_non_stopwords)
train_feats_no_stopwords, test_feats_no_stopwords = split_label_feats(lfeats_without_stopwords, split=0.75)

print("Количество обучающих отзывов без использования стоп-слов:", len(train_feats_no_stopwords))
print("Количество тестовых отзывов без использования стоп-слов:", len(test_feats_no_stopwords))

# Обучение классификатора
nb_classifier_without_stopwords = NaiveBayesClassifier.train(train_feats_no_stopwords)

acc_val_without_stopwords = accuracy(nb_classifier_without_stopwords, test_feats_no_stopwords)
print(f"Точность классификатора без использования стоп-слов: {acc_val_without_stopwords:.2f}")

print("Слова с наибольшим влиянием на классификацию (без использования стоп-слов):")
nb_classifier_without_stopwords.show_most_informative_features(10)
