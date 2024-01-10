import ast

import pandas as pd

results = pd.read_csv('results/scores.csv')

results_log_reg = results[results['pipeline_name'].str.contains('LogisticRegression')]

results_log_reg[' pipeline_scores'] = results_log_reg[' pipeline_scores'].apply(lambda lst: ast.literal_eval(lst))

# highest score

results_log_reg['Score'] = results_log_reg[' pipeline_scores'].apply(lambda x: float([s.split(': ')[1] for s in x if s.startswith('Score')][0]))

index_of_max_score = results_log_reg['Score'].idxmax()

pipeline_highest_score_log_reg = results_log_reg.loc[index_of_max_score, 'pipeline_name']

highest_score_info_log_reg = [pipeline_highest_score_log_reg, results_log_reg['Score'].max()]


# highest precision

results_log_reg['precision'] = results_log_reg[' pipeline_scores'].apply(lambda x: float([s.split(': ')[1] for s in x if s.startswith('precision')][0]))

index_of_max_precision = results_log_reg['precision'].idxmax()

pipeline_highest_precision_log_reg = results_log_reg.loc[index_of_max_precision, 'pipeline_name']

highest_precision_info_log_reg = [pipeline_highest_precision_log_reg, results_log_reg['precision'].max()]


# highest recall 

results_log_reg['Recall'] = results_log_reg[' pipeline_scores'].apply(lambda x: float([s.split(': ')[1] for s in x if s.startswith('Recall')][0]))

index_of_max_Recall = results_log_reg['Recall'].idxmax()

pipeline_highest_Recall_log_reg = results_log_reg.loc[index_of_max_Recall, 'pipeline_name']

highest_Recall_info_log_reg = [pipeline_highest_Recall_log_reg, results_log_reg['Recall'].max()]


# highest roc auc

results_log_reg['ROC AUC'] = results_log_reg[' pipeline_scores'].apply(lambda x: float([s.split(': ')[1] for s in x if s.startswith('ROC AUC')][0]))

index_of_max_ROC_AUC = results_log_reg['ROC AUC'].idxmax()

pipeline_highest_ROC_AUC_log_reg = results_log_reg.loc[index_of_max_ROC_AUC, 'pipeline_name']

highest_ROC_AUC_info_log_reg = [pipeline_highest_ROC_AUC_log_reg, results_log_reg['ROC AUC'].max()]



results_Rand_Forest_Class = results[results['pipeline_name'].str.contains('RandomForestClassifier')]

results_Rand_Forest_Class[' pipeline_scores'] = results_Rand_Forest_Class[' pipeline_scores'].apply(lambda lst: ast.literal_eval(lst))

# highest score

results_Rand_Forest_Class['Score'] = results_Rand_Forest_Class[' pipeline_scores'].apply(lambda x: float([s.split(': ')[1] for s in x if s.startswith('Score')][0]))

index_of_max_score = results_Rand_Forest_Class['Score'].idxmax()

pipeline_highest_score_Rand_Forest_Clas = results_Rand_Forest_Class.loc[index_of_max_score, 'pipeline_name']

highest_score_info_Rand_Forest_Class = [pipeline_highest_score_Rand_Forest_Clas, results_Rand_Forest_Class['Score'].max()]


# highest precision

results_Rand_Forest_Class['precision'] = results_Rand_Forest_Class[' pipeline_scores'].apply(lambda x: float([s.split(': ')[1] for s in x if s.startswith('precision')][0]))

index_of_max_precision = results_Rand_Forest_Class['precision'].idxmax()

pipeline_highest_precision_Rand_Forest_Clas = results_Rand_Forest_Class.loc[index_of_max_precision, 'pipeline_name']

highest_precision_info_Rand_Forest_Class = [pipeline_highest_precision_Rand_Forest_Clas, results_Rand_Forest_Class['precision'].max()]


# highest recall 

results_Rand_Forest_Class['Recall'] = results_Rand_Forest_Class[' pipeline_scores'].apply(lambda x: float([s.split(': ')[1] for s in x if s.startswith('Recall')][0]))

index_of_max_Recall = results_Rand_Forest_Class['Recall'].idxmax()

pipeline_highest_Recall_Rand_Forest_Clas = results_Rand_Forest_Class.loc[index_of_max_Recall, 'pipeline_name']

highest_Recall_info_Rand_Forest_Class = [pipeline_highest_Recall_Rand_Forest_Clas, results_Rand_Forest_Class['Recall'].max()]


# highest roc auc

results_Rand_Forest_Class['ROC AUC'] = results_Rand_Forest_Class[' pipeline_scores'].apply(lambda x: float([s.split(': ')[1] for s in x if s.startswith('ROC AUC')][0]))

index_of_max_ROC_AUC = results_Rand_Forest_Class['ROC AUC'].idxmax()

pipeline_highest_ROC_AUC_Rand_Forest_Clas = results_Rand_Forest_Class.loc[index_of_max_ROC_AUC, 'pipeline_name']

highest_ROC_AUC_info_Rand_Forest_Class = [pipeline_highest_ROC_AUC_Rand_Forest_Clas, results_Rand_Forest_Class['ROC AUC'].max()]


results_SVC = results[results['pipeline_name'].str.contains('SVC')]

results_SVC[' pipeline_scores'] = results_SVC[' pipeline_scores'].apply(lambda lst: ast.literal_eval(lst))

# highest score

results_SVC['Score'] = results_SVC[' pipeline_scores'].apply(lambda x: float([s.split(': ')[1] for s in x if s.startswith('Score')][0]))

index_of_max_score = results_SVC['Score'].idxmax()

pipeline_highest_score_SVC = results_SVC.loc[index_of_max_score, 'pipeline_name']

highest_score_info_SVC = [pipeline_highest_score_SVC, results_SVC['Score'].max()]

# highest precision

results_SVC['precision'] = results_SVC[' pipeline_scores'].apply(lambda x: float([s.split(': ')[1] for s in x if s.startswith('precision')][0]))

index_of_max_precision = results_SVC['precision'].idxmax()

pipeline_highest_precision_SVC = results_SVC.loc[index_of_max_precision, 'pipeline_name']

highest_precision_info_SVC = [pipeline_highest_precision_SVC, results_SVC['precision'].max()]

# highest recall 

results_SVC['Recall'] = results_SVC[' pipeline_scores'].apply(lambda x: float([s.split(': ')[1] for s in x if s.startswith('Recall')][0]))

index_of_max_Recall = results_SVC['Recall'].idxmax()

pipeline_highest_Recall_SVC = results_SVC.loc[index_of_max_Recall, 'pipeline_name']

highest_Recall_info_SVC = [pipeline_highest_Recall_SVC, results_SVC['Recall'].max()]

# highest roc auc

results_SVC['ROC AUC'] = results_SVC[' pipeline_scores'].apply(lambda x: float([s.split(': ')[1] for s in x if s.startswith('ROC AUC')][0]))

index_of_max_ROC_AUC = results_SVC['ROC AUC'].idxmax()

pipeline_highest_ROC_AUC_SVC = results_SVC.loc[index_of_max_ROC_AUC, 'pipeline_name'], 

highest_ROC_AUC_info_SVC = [pipeline_highest_ROC_AUC_SVC, results_SVC['ROC AUC'].max()]


########## FAST TEXT #############

data = pd.read_csv('results/fasttext_results_raw_model.csv')

# Find the index of the row with the maximum score for each metric
idx_max_accuracy = data['Accuracy'].idxmax()
idx_max_precision = data['Precision'].idxmax()
idx_max_recall = data['Recall'].idxmax()
idx_max_f1 = data['F1'].idxmax()
idx_max_roc_auc = data['ROC_AUC'].idxmax()

# Create a DataFrame containing the rows with the maximum scores for each metric
best_results = pd.DataFrame([
    data.loc[idx_max_accuracy],
    data.loc[idx_max_precision],
    data.loc[idx_max_recall],
    data.loc[idx_max_f1],
    data.loc[idx_max_roc_auc]
])

best_results = best_results[['Preprocessing flow', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']]