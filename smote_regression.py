#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import numpy as np
import pickle
import argparse
import filter_func
import os
import multiprocessing


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--db_path', type=str, default='db/colon.csv', help='Path to the dataset')
parser.add_argument('--iter', type=int, default=1000, help='Number of iterations')
parser.add_argument('--seed', type=int, default=42, help='Seed for random state')
parser.add_argument('--target_dir', type=str, default='./tmp', help='Directory to save the results')
parser.add_argument('--genomics', action='store_true', help='Use genomics features', default=False)
# Add command line argument to select the model to use between logistic regression, SVM and KNN
parser.add_argument('--model', type=str, default='logistic', help='Model to use: logistic, svm, knn')
args = parser.parse_args()
iterations = args.iter
seed = args.seed
target_dir = args.target_dir
genomics = args.genomics
# parse the model argument
model_name = args.model
if model_name not in ['logistic', 'svm', 'knn']:
    print("Invalid model argument")
    exit(1)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# empty the directory
for f in os.listdir(target_dir):
    if os.path.isfile(os.path.join(target_dir, f)):
        os.remove(os.path.join(target_dir, f))
    else:
        for ff in os.listdir(os.path.join(target_dir, f)):
            os.remove(os.path.join(target_dir, f, ff))
        os.rmdir(os.path.join(target_dir, f))

# create directories for roc curves, lasso paths, cross val scores , selected features and models
os.makedirs(target_dir + "/roc_curves")
os.makedirs(target_dir + "/lasso_paths")
os.makedirs(target_dir + "/cross_val_scores")
os.makedirs(target_dir + "/selected_features")
os.makedirs(target_dir + "/models")
os.makedirs(target_dir + "/results")



np.random.seed(seed)
X = pd.read_csv(args.db_path)
Y = X['RISK']
fradiomics  = pickle.load(open("./features_radiomics.p", "rb"))
fbiological = pickle.load(open("./features_biological.p", "rb"))
fgenomics = pickle.load(open("./features_genomics.p", "rb"))
if genomics:
    labels = fradiomics + fbiological + fgenomics
    # select only rows which have "HAS_GENOMICS" == 1
    Y = Y[X['HAS_GENOMICS'] == 1]
    X = X[X['HAS_GENOMICS'] == 1]
else:
    labels = fradiomics + fbiological

X = X[labels]
X = filter_func.process_colon_db(X)
if genomics:
    X = filter_func.process_genomical_features(X, fgenomics)
# fill missing values with the mean
X.fillna(X.mean(), inplace=True)

# function that implements the logistic regression with L1 regularization and cross-validation
def run_iteration(i):
    # globals
    global scores
    global regul_strengths
    global models
    global selected_features_dict

    # print iteration number
    print("Iteration: {}".format(i))
    
    # Set a different seed for each iteration
    np.random.seed(seed + i)
    # Split the dataset into training and test sets
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=seed + i)
    
    # Standardize the data
    scaler = StandardScaler()
    XTrain = scaler.fit_transform(XTrain)
    XTest = scaler.fit_transform(XTest)

    # Apply SMOTE to balance the classes
    sm = SMOTE(random_state=seed + i)
    # Apply SMOTE only on the training set
    XTrain, YTrain = sm.fit_resample(XTrain, YTrain)
    
    # Fit the model
    model = LogisticRegressionCV(cv=5, penalty='l1', solver='saga', max_iter=int(10e6), Cs=100, refit=True)

    model.fit(XTrain, YTrain)
    
    # Get the coefficients of the model
    B = model.coef_
    B0 = model.intercept_
    # Find indices of significant features
    mincoefs = list(filter(lambda i: B[0][i] != 0, range(len(B[0]))))
    lab = [labels[x] for x in mincoefs]
    # save the selected features in a text file
    with open(target_dir + "/selected_features/selected_features_{}.txt".format(i), "w") as f:
        for l in lab:
            f.write(l + "\n")
    plt.figure()
    #plt.plot(np.log(model.Cs_), model.coefs_paths_[1][4])
    plt.plot(model.Cs_, model.coefs_paths_[1][4])
    #plt.axvline(np.log(model.C_), color='green', linestyle=':', linewidth=0.5)
    plt.axvline(model.C_, color='green', linestyle=':', linewidth=0.5)
    plt.xscale('log')
    # plt.legend(lab)
    #plt.show()
    plt.savefig(target_dir + "/lasso_paths/lasso_path_{}.png".format(i))
    plt.close()
    # Plot cross-validated scores
    mean_cv_scores = np.mean(model.scores_[1], axis=0)  # mean across folds for class 1
    regularization_strengths = model.Cs_
    plt.figure()
    plt.plot(regularization_strengths, mean_cv_scores, marker='o', linestyle='--', color='blue')
    #plt.axvline(np.log(model.C_), color='green', linestyle=':', linewidth=0.5)
    plt.axvline(model.C_, color='green', linestyle=':', linewidth=0.5)
    plt.xscale('log')
    plt.xlabel('Regularization Strength (C)')
    plt.ylabel('Mean Cross-Validated Score')
    plt.title('Cross-Validated Score vs. Regularization Strength (L1)')
    plt.grid(True)
    plt.savefig(target_dir + "/cross_val_scores/cross_val_score_{}.png".format(i))
    plt.close()
    
    YPred = model.predict_proba(XTest)[:, 1]
    # Predict labels
    YPredLab = model.predict(XTest) 
    
    fpr, tpr, thresholds = roc_curve(YTest, YPred)
    auc_score = roc_auc_score(YTest, YPred)

    # Compute precision, recall, f1-score, accuracy and specificity for class 1
    precision = np.sum((YPredLab == 1) & (YTest == 1)) / np.sum(YPredLab == 1)
    recall = np.sum((YPredLab == 1) & (YTest == 1)) / np.sum(YTest == 1)
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy = np.sum(YPredLab == YTest) / len(YTest)
    specificity = np.sum((YPredLab == 0) & (YTest == 0)) / np.sum(YTest == 0)


    # Compute precision, recall, f1-score for class 0
    if np.sum(YPredLab == 0) == 0:
        precision0 = 0
        recall0 = 0
        f1_score0 = 0
    else:
        precision0 = np.sum((YPredLab == 0) & (YTest == 0)) / np.sum(YPredLab == 0)
        recall0 = np.sum((YPredLab == 0) & (YTest == 0)) / np.sum(YTest == 0)
        f1_score0 = 2 * precision0 * recall0 / (precision0 + recall0)





    print("AUC score for iteration {}: {}".format(i, auc_score))
    
    # Step 6: Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(target_dir + "/roc_curves/roc_curve_{}.png".format(i))
    plt.close()
    
    


    # store model
    pickle.dump(model, open(target_dir + "/models/model_{}.p".format(i), "wb"))

    # save model AUC score, selected features, selected regularization strength, model, precision, recall, f1-score
    with open(target_dir + "/results/results_{}.txt".format(i), "w") as f:
        f.write("AUC score: {}\n".format(auc_score))
        f.write("Precision: {}\n".format(precision))
        f.write("Recall: {}\n".format(recall))
        f.write("Specificity: {}\n".format(specificity))
        f.write("Accuracy: {}\n".format(accuracy))
        f.write("F1-score: {}\n".format(f1_score))
        f.write("Precision class 0: {}\n".format(precision0))
        f.write("Recall class 0: {}\n".format(recall0))
        f.write("F1-score class 0: {}\n".format(f1_score0))
        f.write("Selected features:\n")
        for l in lab:
            f.write("{}\n".format(l))
        f.write("Selected regularization strength: {}\n".format(model.C_))
        f.write("Inverse of regularization strength: {}\n".format(1/model.C_))
        f.write("Array of regularization strengths: {}\n".format(model.Cs_))
        f.write("Array of scores: {}\n".format(model.scores_[1][0])) 

    # return auc score, selected features, selected regularization strength, model, precision, recall, f1-score, iteration
    return auc_score, lab, model.C_, model, precision, recall, f1_score, precision0, recall0, f1_score0, specificity, accuracy, i




# run iterations in parallel
pool = multiprocessing.Pool()
results = pool.map(run_iteration, range(iterations))

# extract the results
scores = [r[0] for r in results]
selected_features = [r[1] for r in results]
regul_strengths = [r[2] for r in results]
models = [r[3] for r in results]
precisions = [r[4] for r in results]
recalls = [r[5] for r in results]
f1_scores = [r[6] for r in results]
precisions0 = [r[7] for r in results]
recalls0 = [r[8] for r in results]
f1_scores0 = [r[9] for r in results]
specificities = [r[10] for r in results]
accuracies = [r[11] for r in results]
iterations_list = [r[12] for r in results]


# produce selected_features_dict with the frequency of each selected feature
selected_features_dict = {}
for sf in selected_features:
    for f in sf:
        if f in selected_features_dict:
            selected_features_dict[f] += 1
        else:
            selected_features_dict[f] = 1


print("Mean AUC score: {}".format(np.mean(scores)))
print("Std AUC score: {}".format(np.std(scores)))
# compute the frequency of each selected feature
selected_features_dict = {k: v/iterations for k, v in selected_features_dict.items()}
# sort the dictionary by frequency
selected_features_dict = dict(sorted(selected_features_dict.items(), key=lambda item: item[1], reverse=True))

# compute the average number of features selected in each iteration
avg_features = np.mean([len(sf) for sf in selected_features])

# save the selected features dictionary in a text file
# save in the same file the mean and std of the AUC scores
# save in the same file the mean and std of the precision, recall, f1-score
# save in the same file the selected regularization strengths mean and std
# save in the same file the simulation parameters
with open(target_dir + "/results.txt", "w") as f:
    f.write("Mean AUC score: {}\n".format(np.mean(scores)))
    f.write("Std AUC score: {}\n".format(np.std(scores)))
    f.write("AUC score confidence interval: [{}, {}]\n".format(np.mean(scores) - 1.96 * np.std(scores), np.mean(scores) + 1.96 * np.std(scores) ))
    f.write("Mean precision: {}\n".format(np.mean(precisions)))
    f.write("Std precision: {}\n".format(np.std(precisions)))
    f.write("Precision confidence interval: [{}, {}]\n".format(np.mean(precisions) - 1.96 * np.std(precisions) , np.mean(precisions) + 1.96 * np.std(precisions) ))
    f.write("Mean recall: {}\n".format(np.mean(recalls)))
    f.write("Std recall: {}\n".format(np.std(recalls)))
    f.write("Recall confidence interval: [{}, {}]\n".format(np.mean(recalls) - 1.96 * np.std(recalls), np.mean(recalls) + 1.96 * np.std(recalls)))
    f.write("Mean f1-score: {}\n".format(np.mean(f1_scores)))
    f.write("Std f1-score: {}\n".format(np.std(f1_scores)))
    f.write("F1-score confidence interval: [{}, {}]\n".format(np.mean(f1_scores) - 1.96 * np.std(f1_scores), np.mean(f1_scores) + 1.96 * np.std(f1_scores)))
    f.write("Mean specificity: {}\n".format(np.mean(specificities)))
    f.write("Std specificity: {}\n".format(np.std(specificities)))
    f.write("Specificity confidence interval: [{}, {}]\n".format(np.mean(specificities) - 1.96 * np.std(specificities), np.mean(specificities) + 1.96 * np.std(specificities)))
    f.write("Mean accuracy: {}\n".format(np.mean(accuracies)))
    f.write("Std accuracy: {}\n".format(np.std(accuracies)))
    f.write("Accuracy confidence interval: [{}, {}]\n".format(np.mean(accuracies) - 1.96 * np.std(accuracies), np.mean(accuracies) + 1.96 * np.std(accuracies) ))
    f.write("Mean precision class 0: {}\n".format(np.mean(precisions0)))
    f.write("Std precision class 0: {}\n".format(np.std(precisions0)))
    f.write("Precision confidence interval class 0: [{}, {}]\n".format(np.mean(precisions0) - 1.96 * np.std(precisions0) , np.mean(precisions0) + 1.96 * np.std(precisions0)))
    f.write("Mean recall class 0: {}\n".format(np.mean(recalls0)))
    f.write("Std recall class 0: {}\n".format(np.std(recalls0)))
    f.write("Recall confidence interval class 0: [{}, {}]\n".format(np.mean(recalls0) - 1.96 * np.std(recalls0), np.mean(recalls0) + 1.96 * np.std(recalls0)))
    f.write("Mean f1-score class 0: {}\n".format(np.mean(f1_scores0)))
    f.write("Std f1-score class 0: {}\n".format(np.std(f1_scores0)))
    f.write("F1-score confidence interval class 0: [{}, {}]\n".format(np.mean(f1_scores0) - 1.96 * np.std(f1_scores0), np.mean(f1_scores0) + 1.96 * np.std(f1_scores0)))
    f.write("Mean regularization strength: {}\n".format(np.mean(regul_strengths)))
    f.write("Std regularization strength: {}\n".format(np.std(regul_strengths)))
    f.write("Regularization strength confidence interval: [{}, {}]\n".format(np.mean(regul_strengths) - 1.96 * np.std(regul_strengths), np.mean(regul_strengths) + 1.96 * np.std(regul_strengths)))
    f.write("Array of regularization strengths: {}\n".format(regul_strengths))
    f.write("Array of scores: {}\n".format(scores))
    f.write("Simulation parameters:\n")
    f.write("Iterations: {}\n".format(iterations))
    f.write("Seed: {}\n".format(seed))
    f.write("Avg. # Selected features: {}\n".format(avg_features))
    f.write("Selected features:\n")
    for k, v in selected_features_dict.items():
        f.write("{}: {}\n".format(k, v))


# save in a csv file the 
# save the models in a pickle file the AUC scores, precision, recall and f1-score of each iteration
with open(target_dir + "/results.csv", "w") as f:
    # save data in a list of lists and sort by decreasing AUC
    data = []
    for i in range(len(scores)):
        data.append(["model_{}".format(iterations_list[i]), scores[i], precisions[i], recalls[i], f1_scores[i]])
    data = sorted(data, key=lambda x: x[1], reverse=True)
    # write the header
    f.write("Model, AUC, Precision, Recall, F1-score\n")
    for i in range(len(scores)):
        f.write("{}, {}, {}, {}, {}\n".format(data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]))

pickle.dump(models, open(target_dir + "/models.p", "wb"))
# save simulation configuration in a text file insided the target directory
with open(target_dir + "/config.txt", "w") as f:
    f.write("Model: {}\n".format(model_name))
    f.write("Iterations: {}\n".format(iterations))
    f.write("Seed: {}\n".format(seed))
    f.write("Colon risk dataset\n")
    # save used dataset
    f.write("Dataset: {}\n".format(args.db_path))
