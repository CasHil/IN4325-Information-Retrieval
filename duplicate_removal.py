import pandas as pd
import nltk
from typing import Tuple

from progress_bar import print_progress_bar

for dependency in ("wordnet", "stopwords"):
    nltk.download(dependency)

rows = 0

def normalizer(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    from normalise import tokenize_basic, normalise, rejoin
    for column in columns:
        df[column] = df[column].apply(lambda question: rejoin(normalise(question, tokenizer=tokenize_basic, verbose=True)))
    return df

def convert_to_lowercase(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = df[column].apply(lambda question: str(question).lower())
    return df

def remove_non_alphanumeric_characters(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = df[column].apply(lambda question: ' '.join(filter(str.isalnum, str(question).split())))
    return df

def filter_stop_words(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    from nltk.corpus import stopwords
    english_stop_words = stopwords.words('english')
    
    for column in columns:
        df[column] = df[column].apply(lambda question: ' '.join([word for word in str(question).split() if word not in english_stop_words]))
    return df

def lemmatize_questions(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    for column in columns:
        df[column] = df[column].apply(lambda question: ' '.join([lemmatizer.lemmatize(word) for word in str(question).split()]))
    return df

def levenshtein_distance(df: pd.DataFrame, columns: Tuple[str, str]) -> pd.DataFrame:
    from Levenshtein import distance
    
    print('Calculating Levenshtein distances')
    
    df['levenshtein_distance'] = pd.Series(dtype='int')
    for row_index, row in df.iterrows():
        df.at[row_index, 'levenshtein_distance'] = distance(row[columns[0]], row[columns[1]])
        
        print_progress_bar(row_index + 1, rows)
        
    return df

def sent2vec(df: pd.DataFrame, columns: Tuple[str, str]) -> pd.DataFrame:
    from scipy import spatial
    from sent2vec.vectorizer import Vectorizer
    vectorizer = Vectorizer()
    print('Calculating sent2vec')
    
    df['cosine_distance'] = pd.Series(dtype='float')
    
    for row_index, row in df.iterrows():
        if not pd.isnull(df.loc[row_index, 'cosine_distance']):
            continue
        question_1 = row[columns[0]]
        question_2 = row[columns[1]]
        vectorizer.run([question_1, question_2])
        vectors = [vectorizer.vectors[row_index * 2], vectorizer.vectors[row_index * 2 + 1]]        
        df.at[row_index, 'cosine_distance'] = spatial.distance.cosine(vectors[0], vectors[1])
        
        print_progress_bar(row_index + 1, rows)
    return df

def common_words(df: pd.DataFrame, columns: Tuple[str, str]) -> pd.DataFrame:
    df['common_words'] = data.apply(lambda row: len(set(str(row[columns[0]]).lower().split()).intersection(set(str(row[columns[1]]).lower().split()))), axis=1)
    return df

if __name__ == "__main__":
    
    
    # Normalize data
    question_columns = ['question1', 'question2']
    
    from pathlib import Path
    
    if not Path('./Annotated development set.csv').is_file():
        # Parse data and remove unnecessary data
        data = pd.read_csv('data/Development set.csv')
        data = data.drop(['id', 'qid1', 'qid2'], axis=1)
    
        rows = len(data.index)
        
        # For testing only
        # data = data.head(100)
        # data = data.iloc[[3]]
        
        data = convert_to_lowercase(data, question_columns)
        data = remove_non_alphanumeric_characters(data, question_columns)
        data = filter_stop_words(data, question_columns)
        data = lemmatize_questions(data, question_columns)
        
        # Calculate similarity
        question_tuple = ('question1', 'question2')
        
        data = levenshtein_distance(data, question_tuple)
        data = sent2vec(data, question_tuple)
        data = common_words(data, question_tuple)
        data.to_csv('Annotated development set.csv')
        
    if not Path('./Annotated test set.csv').is_file():
        # Parse data and remove unnecessary data
        data = pd.read_csv('data/Test set.csv')
        data = data.drop(['qid1', 'qid2', '?'], axis=1)
    
        rows = len(data.index)
        
        # For testing only
        # data = data.head(100)
        # data = data.iloc[[3]]
        
        data = convert_to_lowercase(data, question_columns)
        data = remove_non_alphanumeric_characters(data, question_columns)
        data = filter_stop_words(data, question_columns)
        data = lemmatize_questions(data, question_columns)
        
        # Calculate similarity
        question_tuple = ('question1', 'question2')
        
        data = levenshtein_distance(data, question_tuple)
        data = sent2vec(data, question_tuple)
        data = common_words(data, question_tuple)
        data.to_csv('Annotated development set.csv')
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.metrics import recall_score
    
    df = pd.read_csv('Annotated development set.csv')
    data_parameters = df.filter(['levenshtein_distance', 'cosine_distance','common_words'], axis=1)
    data_results = df.filter(['is_duplicate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data_parameters, data_results, test_size=0.2)
    
    df = pd.read_csv('Annotated test set.csv')
    data_parameters_test = df.filter(['levenshtein_distance', 'cosine_distance','common_words'], axis=1)
    df = df.drop(['question1', 'question2', 'levenshtein_distance', 'cosine_distance', 'common_words'], axis=1)
    del df[df.columns[0]]
    
    def run_classifier(classifier: RandomForestClassifier | DecisionTreeClassifier | KNeighborsClassifier | LogisticRegression):
        classifier_name = classifier.__class__.__name__
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        tpr = recall_score(y_test, y_pred)
        tnr = recall_score(y_test, y_pred, pos_label = 0) 
        fpr = 1 - tnr
        fnr = 1 - tpr
        print(f'TPR {classifier_name}: {tpr}')
        print(f'TNR {classifier_name}: {tnr}')
        print(f'FPR {classifier_name}: {fpr}')
        print(f'FNR {classifier_name}: {fnr}')
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy {classifier_name}: {accuracy}')
        
        df['is_duplicate'] = classifier.predict(data_parameters_test)
        df.to_csv(f'classifier_results/data/{classifier_name}.csv', index=False)
    
    run_classifier(RandomForestClassifier(n_estimators=100))
    run_classifier(DecisionTreeClassifier())
    run_classifier(KNeighborsClassifier())
    run_classifier(LogisticRegression())
