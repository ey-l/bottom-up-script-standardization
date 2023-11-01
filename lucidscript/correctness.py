from logging import warning
import sys
sys.path.append('/Users/eylai/Projects/lucid-script/')
sys.path.append(sys.path[0] + '/..')
from lucidscript.imports import *
import warnings
warnings.filterwarnings("ignore")

# Specific to this script
from sklearn.metrics import jaccard_score
from scipy.stats import wasserstein_distance
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def table_jaccard(before, after):
    '''
    Take before and after dataframe, compute jaccard score:
        * assume same number of rows
        * don't know how to handle new/different column names in the after csv
        * NaN's are considered. the way it's done now is to fill NaN's with max int
        * sklearn's implementation doesn't allow continuous values, so floats are cast into str
        * score is computed for each column, then average
        * perfect score is 1

    :return: float
    '''
    max_int = sys.maxsize
    
    # Prep the dataframes
    before = before.fillna(max_int)
    after = after.fillna(max_int)
    
    for col in before.columns:
        if before[col].dtype == 'float':
            before[col] = before[col].astype(str)
    for col in after.columns:
        if after[col].dtype == 'float':
            after[col] = after[col].astype(str)
    
    # Compute score
    scores = []
    # Take the intersection of before and after columns
    intersect = list(set(before.columns) & set(after.columns))
    before_cols = list(set(before.columns) - set(intersect))
    after_cols = list(set(after.columns) - set(intersect))

    for col in intersect:
        if before[col].dtype == after[col].dtype:
            x = list(before[col].values)
            y = list(after[col].values)
            score = jaccard_score(x, y, average='weighted')
        else:
            score = 0
        scores.append(score)
    
    # For columns unique to each dataframe
    scores += [0]*len(before_cols)
    scores += [0]*len(after_cols)
    
    # Average columns
    return np.array(scores).mean()

def table_EMD(before, after):
    '''
    Take before and after dataframe, compute EMD score:
        * only consider numerical columns for now
        * perfect score is 0
        * for str columns, consider some word embedding such as this tutorial 
        https://pmbaumgartner.github.io/blog/word-movers-distance-exploration/

    :return: float
    '''
    max_int = sys.maxsize
    
    # Prep the dataframes
    before = before.fillna(max_int)
    after = after.fillna(max_int)
    
    # Compute score
    scores = []

    for col in before.columns:
        if before[col].dtype in ['float', 'int']:
            x = list(before[col].values)
            y = list(after[col].values)
            score = wasserstein_distance(x, y)
            scores.append(score)
    
    # Average columns
    return np.array(scores).mean()

def df_accuracy_score(df, model, dataset):
    '''
    Compute the model accuracy with the given dataframe and model of choice
    :param dependent (str): the dependent variable
    '''
    # Lookup for target column
    ys = {'titanic': 'Survived', 
          'competitive-data-science-predict-future-sales': '2015-10',
          'house-prices-advanced-regression-techniques': 'SalePrice',
          'nlp-getting-started': 'target',
          'spaceship-titanic': 'Transported',
          'digit-recognizer': 'label'}
    
    # 'uciml_pima-indians-diabetes-database' doesn't have a named target column
    if dataset not in ys:
        X = df.iloc[:, 0:8]
        y = df.iloc[:, 8]
    else:
        print(df)
        dependent = ys[dataset]
        X = df.drop([dependent], axis = 1)
        y = df[dependent]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                    random_state=0)
    #model = tree.DecisionTreeClassifier()
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if dataset == 'titanic':
        return accuracy_score(y_test, y_pred)
    elif dataset == 'competitive-data-science-predict-future-sales':
        return mean_absolute_error(y_test, y_pred)
    elif dataset == 'house-prices-advanced-regression-techniques':
        return mean_absolute_error(y_test, y_pred)
    elif dataset == 'nlp-getting-started':
        return accuracy_score(y_test, y_pred)
    elif dataset == 'spaceship-titanic':
        return accuracy_score(y_test, y_pred)
    elif dataset == 'digit-recognizer':
        return accuracy_score(y_test, y_pred)
    elif dataset == 'uciml_pima-indians-diabetes-database':
        return accuracy_score(y_test, y_pred)

def model_performance(dataset, before, after):
    '''
    Compute normalized percentage difference:
        * titanic is a classification problem
        * decision tree for now
        * other models: logistics regression, SVM
        * logistics regression and SVM don't like continuous variables 
        * normalized percentage difference between the before vs. after dataframe

    :return score (float): 
    '''
    # Handle missing values
    before = before.fillna(0)
    after = after.fillna(0)
    
    #model = svm.SVC()
    #model = tree.DecisionTreeClassifier(max_depth=3)
    if dataset == 'titanic':
        model = LogisticRegression() #random_state=42, max_iter=100
        for col in before.columns:
            if before[col].dtype == 'float':
                before[col] = before[col].astype(str)
        for col in after.columns:
            if after[col].dtype == 'float':
                after[col] = after[col].astype(str)

    elif dataset == 'competitive-data-science-predict-future-sales':
        model = LinearRegression()

    elif dataset == 'house-prices-advanced-regression-techniques':
        model = LinearRegression()

    elif dataset == 'nlp-getting-started':
        model = LogisticRegression()

    elif dataset == 'spaceship-titanic':
        model = LogisticRegression()

    elif dataset == 'digit-recognizer':
        model = KNeighborsClassifier()

    elif dataset == 'uciml_pima-indians-diabetes-database':
        model = KNeighborsClassifier()

    else:
        model = None
        print("No model for this dataset")

    # Get dummies for categorical columns
    before = pd.get_dummies(before, drop_first=True)
    after = pd.get_dummies(after, drop_first=True)

    before_score = df_accuracy_score(before, model,dataset)
    after_score = df_accuracy_score(after, model,dataset)
    
    score = (before_score - after_score) / before_score
    
    return score

def compute_corr_measures(dataset, data_fp, filenames, corr_fp, i, updated_dataframes):
    row = [i]
    result = {}
    for df in updated_dataframes:
        fn = df[0]
        #print(fn)
        updated = df[1]
        result.setdefault(fn, [])
        try:
            data = pd.read_csv(data_fp + "/" + fn + ".csv")
            result[fn] = compute_corr_measures_per_df(dataset, data, updated)
            #print(result[fn])
        except Exception as e:
            print(e)
            # Semantically incorrect scripts 
            result[fn] = [None, None, None]
    
    for fn in filenames:
        try:
            row.extend(result[fn])
        except:
            row.extend([None, None, None])
    print(row)    
    with open(corr_fp, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def compute_corr_measures_per_df(dataset, data, updated):
    row = []
    row.append(table_jaccard(data, updated))
    row.append(table_EMD(data, updated))
    try:
        row.append(model_performance(dataset, data, updated))
    except Exception as e:
        print(e) 
        row.append(None)
    return row

if __name__ == "__main__":
    """
    Use case
    """
    dataset = 'uciml_pima-indians-diabetes-database' #'spaceship-titanic' #'nlp-getting-started' #"house-prices-advanced-regression-techniques" #"competitive-data-science-predict-future-sales"
    data_path = os.path.join("/Users/eylai/Projects/lucid-script/data/input", dataset)
    before = pd.read_csv(os.path.join(data_path, 'diabetes.csv'), low_memory=False) #'train.csv'

    #table_jaccard(before, after) # => [0,1]
    score = model_performance(dataset, before, before)
    print(score)
