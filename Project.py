import pandas as pd
from math import log
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score 
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import string 
from nltk.corpus import stopwords
from nltk import ngrams



def get_punctuation_free_data_frame(data_frame):
    string_punctuation = string.punctuation
    for i in range(9):
        string_punctuation += str(i)
    for key, rows in data_frame.iterrows():
        input_string = data_frame.loc[key, "title"]
        data_frame.loc[key, "title"] = ''.join([i for i in input_string if i not in string_punctuation])
    return data_frame

def get_stop_word_free_data_frame(data_frame):
    stop_words = list(set(stopwords.words("english")))
    for key, rows in data_frame.iterrows():
        input_string_list = data_frame.loc[key, "title"].split(" ")
        data_frame.loc[key, "title"] = ' '.join([i for i in input_string_list if i.lower() not in stop_words])
    return data_frame


def get_category_wise_word_probabilty(data_frame):
    tags = data_frame["tags"]
    frequency_distribution = {}
    for i in tags:
        if i not in frequency_distribution:
            frequency_distribution[i] = {}

    vocabulary = set()
    for key, rows in data_frame.iterrows():
        title = rows["title"].split()
        for word in title:
            if word not in vocabulary:
                vocabulary.add(word)
            if word not in frequency_distribution[rows["tags"]]:
                frequency_distribution[rows["tags"]][word] = 1
            else:
                frequency_distribution[rows["tags"]][word] += 1


    probability_distribution = {}
    for word in vocabulary:
        for key, value in frequency_distribution.items():
            if key not in probability_distribution:
                probability_distribution[key] = {}
            if word in frequency_distribution[key]:
                # Laplace smoothing Add 1
                probability_distribution[key][word] = log((frequency_distribution[key][word] + 1) / 
                    (len(frequency_distribution[key]) + len(vocabulary)))
            else:
                probability_distribution[key][word] = log(1 / (len(frequency_distribution[key]) + len(vocabulary)))
    return probability_distribution, vocabulary


def predict(test_data_frame, probability_distribution, vocabulary):
    
    correct_prediction = 0
    for key, rows in test_data_frame.iterrows():
        final_probability = {'php': 0, 'c#': 0, 'javascript': 0, 'java': 0, 'ruby-on-rails': 0, 'c++': 0, 'python': 0, 'c': 0}
        question_token = rows["title"].split()
        for token in question_token:
            if token not in vocabulary:
                pass
            else:
                for language in list(final_probability.keys()):
                    final_probability[language] += probability_distribution[language][token]
    
        predicted_value = ''
        max_value = float("-inf")
        for sub_key, sub_value in final_probability.items():
            if sub_value > max_value:
                max_value = sub_value
                predicted_value = sub_key
        if predicted_value == rows["tags"]:
            correct_prediction += 1
    print("Naive Bayes Classifier Non Library Accuracy: ",correct_prediction, "%")

def naive_bayes_classifier_library(data_frame, test_data_frame):
    X_train, X_test, y_train, y_test = train_test_split(data_frame['title'], data_frame['tags'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    correct_prediction = 0
    for key, rows  in test_data_frame.iterrows():
        if clf.predict(count_vect.transform(["i"])) == rows["tags"]:
            correct_prediction += 1
    print("Naive Bayes Classifier Library Accuracy: ",correct_prediction, "%")


def naive_bayes_classifier(data_frame):
    probability_distribution, vocabulary  = get_category_wise_word_probabilty(data_frame)
    test_data_frame = data_frame[:100]
    predict(test_data_frame, probability_distribution, vocabulary)
    naive_bayes_classifier_library(data_frame, test_data_frame)


def svm_classifier(stop_word_free_data):
    title = [i for i in stop_word_free_data.title]
    vectorizer = TfidfVectorizer()
    title_fitted = vectorizer.fit(title)
    title_vectors = [vectorizer.transform([i]) for i in title]
    X_transformed = vectorizer.fit_transform(title)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, stop_word_free_data.tags, test_size = 0.25)
    svclassifier = SVC(kernel='linear')  
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)  
    print("SVM ACCURACY:",accuracy_score(y_test, y_pred) * 100) 
    # print(confusion_matrix(y_test,y_pred))  
    # print(classification_report(y_test,y_pred))


def artificial_neural_network(data_frame, dimension_value):
    language_number = {'php': 0, 'c#': 1, 'javascript': 2, 'java': 3, 'ruby-on-rails': 4, 'c++': 5, 'python': 6, 'c': 7}
    for key, rows in data_frame.iterrows():
        data_frame.loc[key, "tags"] = language_number[rows["tags"]]
    column_attribute_name = [i for i in range(dimension_value)]
    X = data_frame[column_attribute_name]
    Y = data_frame["tags"]
    # create model
    Y = to_categorical(Y)
    model = Sequential()
    model.add(Dense(24, input_dim=dimension_value, activation='relu'))
    model.add(Dense(16, activation='softmax'))
    model.add(Dense(8, activation='softmax'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X, Y, epochs = 200, batch_size=10)
    # evaluate the model
    scores = model.evaluate(X, Y)
    return (model.metrics_names[1], scores[1]*100)

def neural_network(data_frame):
    # Vector size
    dimension_value = 50
    titles = []
    for i in range(dimension_value):
        data_frame[i] = 0
    
    # Getting the word2vec model
    for key, rows in data_frame.iterrows():
        titles.append(data_frame.loc[key, "title"].split(" "))
    model = Word2Vec(titles, min_count=1, size = dimension_value)
    
    # Getting the vector representation of the each title by using the model , getting the 
    # vector representation of it, and finally adding it
    word2vector = []
    for title in titles:
        temp = []
        for word in title:
            temp.append(model[word])
        word2vector.append([sum(i)/len(i) for i in zip(*temp)])


    weight_data_frame = pd.DataFrame.from_records(word2vector)
    weight_data_frame["tags"] = data_frame["tags"]
    
    print(artificial_neural_network(data_frame, dimension_value))
            
def get_n_grams(data_frame):
    
    tags = data_frame["tags"]
    frequency_distribution = {}
    probability_distribution = {}
    for i in tags:
        if i not in frequency_distribution:
            frequency_distribution[i] = []    
    
    for key, rows in data_frame.iterrows():
        frequency_distribution[rows["tags"]].append(rows["title"])
    
    for key, value in frequency_distribution.items():
        probability_distribution[key] = {}
        for sentence in value:
            sentence = sentence.split()
            for i in range(len(sentence)-1):
                for j in range(i, len(sentence)):
                    print(sentence[i], sentence[j])
            # for i, w in enumerate(sentence):
            #     print(i, w)

    # print(frequency_distribution)

if __name__ == "__main__":
    data_frame = pd.read_csv("Final_Tags.csv")
    data_frame = data_frame.sample(frac = 1).reset_index(drop = True)
    data_frame = get_punctuation_free_data_frame(data_frame)
    data_frame = get_stop_word_free_data_frame(data_frame)
    
    naive_bayes_classifier(data_frame)
    # svm_classifier(data_frame)
    # print(data_frame)
    # get_n_grams(data_frame)
    # neural_network(data_frame)
    

