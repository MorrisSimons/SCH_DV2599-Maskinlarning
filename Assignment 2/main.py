# Default libraries
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
import scipy.stats as stats
# My models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv('spambase.data', header=None)
X = data.drop(57, axis=1).values  
y = data[57].values

# Define the models
rf_model = RandomForestClassifier()
svm_model = SVC()
KNN_model = KNeighborsClassifier()

results = {}  # Used to store the results for each model
k_folds = 10

# Split the dataset into training and test set
folds = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)


# Run models
for name, model in zip(['Random Forest', 'KNN', 'SVM'], [rf_model, KNN_model, svm_model]):
    accuracies = []
    f1_scores = []
    training_times = []
    
    for train_index, test_index in folds.split(X, y):
        x_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Start training
        start_time = time.time()
        model.fit(x_train, y_train)
        training_time = time.time() - start_time 
        training_time = training_time * 1000
        # Start testing
        predictions = model.predict(X_test)
       
        # Save the results
        if len(predictions) != len(y_test):
            raise ValueError("Number of predictions does not match number of labels.")
        f1_scores.append(f1_score(y_test, predictions))
        accuracies.append(accuracy_score(y_test, predictions))
        training_times.append(training_time)

    results[name] = {'Accuracy': accuracies, 'F1 Score': f1_scores, 'Training Time': training_times}

tests = ['Accuracy', 'F1 Score', 'Training Time']

for test in tests:
    random_forest = results['Random Forest'][f'{test}']
    knn = results['KNN'][f'{test}']
    svm = results['SVM'][f'{test}']

    data = {
        'Random Forest': results['Random Forest'][test],
        'KNN': results['KNN'][test],
        'SVM': results['SVM'][test]
    }
    
    df = pd.DataFrame(data)
    ranked_df = pd.DataFrame(index=range(1, k_folds + 1))

    # Used in calculating step
    for index, row in df.iterrows():
        ranks = stats.rankdata(row)
        ranked_df.loc[index + 1, 'RF rank'] = ranks[0]
        ranked_df.loc[index + 1, 'knn_rank'] = ranks[1]
        ranked_df.loc[index + 1, 'svm_rank'] = ranks[2]

    # Used in avg and mean step and plotting
    results_df = pd.DataFrame({
        'Fold': range(1, k_folds + 1),
        'Random Forest': random_forest,
        'KNN': knn,
        'SVM': svm
    })
    results_df.set_index('Fold', inplace=True)

    # Create a DataFrame for latex
    df = pd.DataFrame({
        'Fold': range(1, k_folds + 1),
        'Random Forest': random_forest,
        'KNN': knn,
        'SVM': svm
    })
    df.set_index('Fold', inplace=True)

    

    # Calculate rankings and append to DataFrame with 3 decimal places
    for index, row in df.iterrows():
        rankings = row.rank(method='min', ascending=False).astype(float)
        df.at[index, 'Random Forest'] = f"{row['Random Forest']:.3f} ({rankings['Random Forest']})"
        df.at[index, 'KNN'] = f"{row['KNN']:.3f} ({rankings['KNN']})"
        df.at[index, 'SVM'] = f"{row['SVM']:.3f} ({rankings['SVM']})"

    # save add standard deviation and avg to df copy and save to latex
    df_copy = df.copy()
    df_copy.loc['Avg'] = ranked_df.mean()
    df_copy.loc['Avg-Score'] = results_df.mean()
    df_copy.loc['Std'] = results_df.std()
    df_copy.to_latex(f"latex_{test}.txt")

    # calculate friedman test
    n = 10  # Number of samples
    k = 3 # Number of algorithms
    sum_squared = sum(ranked_df['RF rank'])**2 + sum(ranked_df['knn_rank'])**2 + sum(ranked_df['svm_rank'])**2
    freidman_stat = (12 / (n * k * (k + 1)) * sum_squared - 3 * n * (k + 1))
    df = n - 1
    critical_value = 6.2
    print("-----------------------------------------")
    print(f"friedmans: {freidman_stat}")
    print("-----------------------------------------")
    # calcultate Nemenyi test
    alpha = 0.05  # Significance level
    df_inf = 3.314 # for alpha = 0.05 table q range table - https://real-statistics.com/statistics-tables/studentized-range-q-table/
    q_alpha = df_inf/np.sqrt(2)
    print(f"q-alpha {q_alpha}")
    CD = q_alpha * np.sqrt((k * (k + 1)) / (6 * n))
    print(f"Critical Difference (CD) for {test}: {CD}")
    print("-----------------------------------------")

    # Hypothesis Testing
    alpha = 0.05  # Significance level
    if critical_value < freidman_stat:
        print(f"For {test}, the null hypothesis is rejected. There are significant differences among the algorithms.")
    else:
        print(f"For {test}, the null hypothesis is not rejected. There are no significant differences among the algorithms.")

    print(f"Test: {test}, freidman_stat: {freidman_stat}, critical_value: {critical_value}, Critical diffrence: {CD}")

    # compare the algorithms with critical difference
    svm_rank = rankings['SVM']
    knn_rank = rankings['KNN']
    rf_rank = rankings['Random Forest']
    algorithms = ['SVM', 'KNN', 'Random Forest']
    ranks = [svm_rank, knn_rank, rf_rank]
    
    for i in range(len(algorithms)):
        for j in range(i+1, len(algorithms)):
            diff = abs(ranks[i] - ranks[j])
            if diff > CD:
                print(f"The difference between {algorithms[i]} and {algorithms[j]} is significant in {test} due to {diff} > {CD}.")
            else:
                print(f"No significant difference between {algorithms[i]} and {algorithms[j]} due to {diff} > {CD} being false.")
