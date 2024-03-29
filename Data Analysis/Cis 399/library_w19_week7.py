import pandas as pd
import numpy as np
from functools import reduce
from types import SimpleNamespace
import random
import matplotlib.pyplot as plt  
import operator

def predictor_case(row, pred, target):
  case_dict = {(0,0): 'true_negative', (1,1): 'true_positive', (0,1): 'false_negative', (1,0): 'false_positive'}
  actual = row[target]
  prediction = row[pred]
  case = case_dict[(prediction, actual)]
  return case

def informedness(cases):
  tp = 0
  if 'true_positive' in cases:
    tp = cases['true_positive']
  tn = 0
  if 'true_negative' in cases:
    tn = cases['true_negative']
  fp = 0
  if 'false_positive' in cases:
    fp = cases['false_positive']
  fn = 0
  if 'false_negative' in cases:
    fn = cases['false_negative']
  if (((tp+fn) == 0) or ((tn+fp) == 0)):
    return -1
  else:
    recall = 1.0*tp/(tp+fn)
    specificity = 1.0*tn/(tn+fp)
    J = (recall + specificity) - 1
    
  return J

def accuracy(cases):
  tp = 0
  if 'true_positive' in cases:
    tp = cases['true_positive']
  tn = 0
  if 'true_negative' in cases:
    tn = cases['true_negative']
  fp = 0
  if 'false_positive' in cases:
    fp = cases['false_positive']
  fn = 0
  if 'false_negative' in cases:
    fn = cases['false_negative']
  if (tp + tn + fp + fn) == 0:
    return 0
  else:
    return (tp + tn)/(tp+tn+fp+fn)

def f1(cases):
  #the heart of the matrix
  tp = 0
  if 'true_positive' in cases:
    tp = cases['true_positive']
  tn = 0
  if 'true_negative' in cases:
    tn = cases['true_negative']
  fp = 0
  if 'false_positive' in cases:
    fp = cases['false_positive']
  fn = 0
  if 'false_negative' in cases:
    fn = cases['false_negative']
	
	#other measures we can derive
  if (((tp+fn) == 0) or ((tp+fp) == 0)):
    return 0
  else:
    recall = 1.0*tp/(tp+fn)
    precision = 1.0*tp/(tp+fp)
	
	#now for the one we want
  if ((recall == 0) or (precision == 0)):
    return 0
  else:
	   f1 = 2/(1/recall + 1/precision)
	
  return f1

def gig(starting_table, split_column, target_column):
    
    #split into two branches, i.e., two sub-tables
    true_table = starting_table.loc[starting_table[split_column] == 1]
    false_table = starting_table.loc[starting_table[split_column] == 0]
    
    #Now see how the target column is divided up in each sub-table (and the starting table)
    true_counts = true_table[target_column].value_counts()  # Note using true_table and not starting_table
    false_counts = false_table[target_column].value_counts()  # Note using false_table and not starting_table
    starting_counts = starting_table[target_column].value_counts() 
    
    #compute the gini impurity for the 3 tables
    starting_gini = gini(starting_counts)
    true_gini = gini(true_counts)
    false_gini = gini(false_counts)

    #compute the weights
    starting_size = len(starting_table.index)
    true_weight = 0.0 if starting_size == 0 else len(true_table.index)/starting_size
    false_weight = 0.0 if starting_size == 0 else len(false_table.index)/starting_size
    
    #wrap it up and put on a bow
    gig = starting_gini - (true_weight * true_gini + false_weight * false_gini)
    
    return gig

def gini(counts):
    (p0,p1) = probabilities(counts)
    sum_probs = p0**2 + p1**2
    gini = 1 - sum_probs
    return gini

def probabilities(counts):
    count_0 = 0 if 0 not in counts else counts[0]  #could have no 0 values
    count_1 = 0 if 1 not in counts else counts[1]
    total = count_0 + count_1
    probs = (0,0) if total == 0 else (count_0/total, count_1/total)  #build 2-tuple
    return probs

def build_pred(column, branch):
    return lambda row: row[column] == branch

def find_best_splitter(table, choice_list, target):
  
    assert (len(table)>0),"Cannot split empty table"
    assert (target in table),"Target must be column in table"
    
    gig_scores = map(lambda col: (col, gig(table, col, target)), choice_list)  #compute tuple (col, gig) for each column
    gig_sorted = sorted(gig_scores, key=lambda item: item[1], reverse=True)  # sort on gig
    return gig_sorted

from functools import reduce

def generate_table(table, conjunction):
  
    assert (len(table)>0),"Cannot generate from empty table"

    sub_table = reduce(lambda subtable, pair: subtable.loc[pair[1]], conjunction, table)
    return sub_table

def compute_prediction(table, target):
  
    assert (len(table)>0),"Cannot predict from empty table"
    assert (target in table),"Target must be column in table"
    
    counts = table[target].value_counts()  # counts looks like {0: v1, 1: v2}

    if 0 not in counts:
        prediction = 1
    elif 1 not in counts:
        prediction = 0
    elif counts[1] > counts[0]:  # ties go to 0 (negative)
        prediction = 1
    else:
        prediction = 0

    return prediction

def build_tree_iter(table, choices, target, hypers={} ):

    assert (len(choices)>0),"Must have at least one column in choices"
    assert (target in table), "Target column not in table"
    assert (len(table) > 1), "Table must have more than 1 row"
    
    k = hypers['max-depth'] if 'max-depth' in hypers else min(4, len(choices))
    gig_cutoff = hypers['gig-cutoff'] if 'gig-cutoff' in hypers else 0.0
    
    def iterative_build(k):
        columns_sorted = find_best_splitter(table, choices, target)
        (best_column, gig_value) = columns_sorted[0]
        
        #Note I add _1 or _0 to make it more readable for debugging
        current_paths = [{'conjunction': [(best_column, build_pred(best_column, 1))],
                          'prediction': None,
                          'gig_score': gig_value},
                         {'conjunction': [(best_column, build_pred(best_column, 0))],
                          'prediction': None,
                          'gig_score': gig_value}
                        ]
        k -= 1  # we just built a level as seed so subtract 1 from k
        tree_paths = []  # add completed paths here
        
        while k>0:
            new_paths = []
            for path in current_paths:
                old_conjunction = path['conjunction']  # a list of (name, lambda)
                before_table = generate_table(table, old_conjunction)  #the subtable the current conjunct leads to
                columns_sorted = find_best_splitter(before_table, choices, target)
                (best_column, gig_value) = columns_sorted[0]
                if gig_value > gig_cutoff:
                    new_path_1 = {'conjunction': old_conjunction + [(best_column, build_pred(best_column, 1))],
                                'prediction': None,
                                 'gig_score': gig_value}
                    new_paths.append( new_path_1 ) #true
                    new_path_0 = {'conjunction': old_conjunction + [(best_column, build_pred(best_column, 0))],
                                'prediction': None,
                                 'gig_score': gig_value}
                    new_paths.append( new_path_0 ) #false
                else:
                    #not worth splitting so complete the path with a prediction
                    path['prediction'] = compute_prediction(before_table, target)
                    tree_paths.append(path)
            #end for loop
            
            current_paths = new_paths
            if current_paths != []:
                k -= 1
            else:
                break  # nothing left to extend so have copied all paths to tree_paths
        #end while loop

        #Generate predictions for all paths that have None
        for path in current_paths:
            conjunction = path['conjunction']
            before_table = generate_table(table, conjunction)
            path['prediction'] = compute_prediction(before_table, target)
            tree_paths.append(path)
        return tree_paths

    return {'paths': iterative_build(k), 'weight': None}

def tree_predictor(row, tree):
    
    #go through each path, one by one (could use a map instead of for loop?)
    for path in tree['paths']:
        conjuncts = path['conjunction']
        result = map(lambda tuple: tuple[1](row), conjuncts)  # potential to be parallelized
        if all(result):
            return path['prediction']
    raise LookupError('No true paths found for row: ' + str(row))

def path_id(row, tree):
	assert (len(tree['paths']) > 0)
	for path in tree['paths']:
		conjuncts = path['conjunction']
		result = map(lambda tuple: tuple[1](row), conjuncts)  # potential to be parallelized
		if all(result):
	  		return tree['paths'].index(path)

def reorder_paths(table, tree):
	path_count = table.apply(lambda row: path_id(row, tree), axis = 1)
	value = path_count.value_counts()
	path = sorted(value.items(), key=lambda x: x[1], reverse=True)
	print(path)
	new_paths = []
	prev_path = tree3['paths']
	for a, b in plist3:
		prev = prev_path[a]
		new_paths.append(prev)
	return new_paths

def produce_scores(table, tree, target):
    scratch_table = pd.DataFrame(columns=['prediction', 'actual'])
    scratch_table['prediction'] = table.apply(lambda row: tree_predictor(row, tree), axis=1)
    scratch_table['actual'] = table[target]  # just copy the target column
    cases = scratch_table.apply(lambda row: predictor_case(row, pred='prediction', target='actual'), axis=1)
    vc = cases.value_counts()
    return [accuracy(vc), f1(vc), informedness(vc)]

def k_fold(table, k, target, hypers, candidate_columns):
  
    #set up the table where we will record fold results
    result_columns = ['name',  'accuracy', 'f1', 'informedness']
    k_fold_results_table = pd.DataFrame(columns=result_columns)
    
    #generate the slices
    total_len = len(table.index)
    slice_size = int(total_len/(1.0*k))
    slices = []
    for i in range(k-1):
        a_slice =  table[i*slice_size:(i+1)*slice_size]
        slices.append( a_slice )
    slices.append( table[(k-1)*slice_size:] )  # whatever is left
    
    #generate test results
    all_scores = []  #keep track of all k results
    for i in range(k):
        test_table = slices[i]
        train_table = compute_training(slices, i)
        fold_tree = build_tree_iter(train_table, candidate_columns, target, hypers)  # train
        scores = produce_scores(test_table, fold_tree, target)  # test
        results_row = {'name': 'fold_'+str(i), 'accuracy': scores[0], 'f1': scores[1], 'informedness': scores[2]}
        k_fold_results_table = k_fold_results_table.append(results_row,ignore_index=True)
        all_scores.append(scores)
    
    #compute average of all folds
    avg_scores = tuple(reduce(lambda total, triple: np.add(triple, total), all_scores)/k)
    results_row = {'name': 'average', 'accuracy': avg_scores[0], 'f1': avg_scores[1], 'informedness': avg_scores[2]}
    k_fold_results_table = k_fold_results_table.append(results_row,ignore_index=True)
    
    #note that I add the meta comment as last step to avoid it being wiped out
    k_fold_results_table.meta = SimpleNamespace()
    k_fold_results_table.meta.hypers  = hypers # adds comment to remind me of hyper params used
    
    return k_fold_results_table

def compute_training(slices, left_out):
    training_slices = []
    for i,slice in enumerate(slices):
        if i == left_out:
            continue
        training_slices.append(slices[i])
    return pd.concat(training_slices)

from sklearn.utils import shuffle

#Determine if slices are mutually exclusive
def verify_unique(slices):
    print(('total length all slices', sum([len(s) for s in slices])))
    for i, a_slice in enumerate(slices[:-1]):
        a_set = set(a_slice.index)
        for j, b_slice in enumerate(slices[i+1:]):
            b_set = set(b_slice.index)
            int_set = a_set.intersection(b_set)  # should be empty set as result
            print((i,j+i+1,int_set))
    return None

def k_fold_random(table, k, target, hypers, candidate_columns):
  #set up the table where we will record fold results
    result_columns = ['name',  'accuracy', 'f1', 'informedness']
    k_fold_results_table = pd.DataFrame(columns=result_columns)
    
    #generate the slices
    # here is sequential slice code from k_fold if you want to use it as base.
    # modify it to produce slices with random rows in each slice.

    table = shuffle(loan_table)
    total_len = len(table.index)
    slice_size = int(total_len/(1.0*k))
    slices = []
    #generate the slices
    for i in range(k-1):
      a_slice =  table[i*slice_size:(i+1)*slice_size]
      slices.append( a_slice )
    slices.append( table[(k-1)*slice_size:] )

    verify_unique(slices)  # I ask you to define this debugging function below
    
    #generate test results
    all_scores = []  #keep track of all k results
    for i in range(k):
        test_table = slices[i]
        train_table = compute_training(slices, i)
        fold_tree = build_tree_iter(train_table, candidate_columns, target, hypers)  # train
        scores = produce_scores(test_table, fold_tree, target)  # test
        results_row = {'name': 'fold_'+str(i), 'accuracy': scores[0], 'f1': scores[1], 'informedness': scores[2]}
        k_fold_results_table = k_fold_results_table.append(results_row,ignore_index=True)
        all_scores.append(scores)
    
    #compute average of all folds
    avg_scores = tuple(reduce(lambda total, triple: np.add(triple, total), all_scores)/5)
    results_row = {'name': 'average', 'accuracy': avg_scores[0], 'f1': avg_scores[1], 'informedness': avg_scores[2]}
    k_fold_results_table = k_fold_results_table.append(results_row,ignore_index=True)
    
    #note that I add the meta comment as last step to avoid it being wiped out
    k_fold_results_table.meta = SimpleNamespace()
    k_fold_results_table.meta.hypers  = hypers # adds comment to remind me of hyper params used
    
    return k_fold_results_table

def vote_taker(row, forest):
    votes = {0:0, 1:0}
    for tree in forest:
        prediction = tree_predictor(row, tree)
        votes[prediction] += 1
    winner = 1 if votes[1]>votes[0] else 0  #ties go to 0
    return winner

def forest_scores(table, forest, target):
    scratch_table = pd.DataFrame(columns=['prediction', 'actual'])
    scratch_table['prediction'] = table.apply(lambda row: vote_taker(row, forest), axis=1)  #only change is to call vote_taker
    scratch_table['actual'] = table[target]  # just copy the target column
    cases = scratch_table.apply(lambda row: predictor_case(row, pred='prediction', target='actual'), axis=1)
    vc = cases.value_counts()
    return [accuracy(vc), f1(vc), informedness(vc)]

def forest_builder(table, column_choices, target, hypers):

    tree_n = 5 if 'total-trees' not in hypers else hypers['total-trees']
    m = int(len(column_choices)**.5) if 'm' not in hypers else hypers['m']
    k = hypers['max-depth'] if 'max-depth' in hypers else min(2, len(column_choices))
    gig_cutoff = hypers['gig-cutoff'] if 'gig-cutoff' in hypers else 0.0
    rgen = hypers['random-state'] if 'random-state' in hypers else 0  #an int will work as seed with the sample method.

    #build a single tree of depth n - call it multiple times to build multiple trees
    def iterative_build(n):
        train = table.sample(frac=1.0, replace=True, random_state=rgen)
        train = train.reset_index()
        left_out = table.loc[~table.index.isin(train['index'])]
        left_out = left_out.reset_index() # this gives us the old index in its own column
        oob_list = left_out['index'].tolist()  # list of row indices from original titanic table
        
        rcols = random.sample(column_choices, m)  # subspcace sampling - uses random.seed, not rng
        columns_sorted = find_best_splitter(train, rcols, target)
        (best_column, gig_value) = columns_sorted[0]

        #Note I add _1 or _0 to make it more readable for debugging
        current_paths = [{'conjunction': [(best_column+'_1', build_pred(best_column, 1))],
                          'prediction': None,
                          'gig_score': gig_value},
                         {'conjunction': [(best_column+'_0', build_pred(best_column, 0))],
                          'prediction': None,
                          'gig_score': gig_value}
                        ]
        n -= 1  # we just built a level as seed so subtract 1 from n
        tree_paths = []  # add completed paths here

        while n>0:
            new_paths = []
            for path in current_paths:
                conjunct = path['conjunction']  # a list of (name, lambda)
                before_table = generate_table(train, conjunct)  #the subtable the current conjunct leads to
                rcols = random.sample(column_choices, m)  # subspace
                columns_sorted = find_best_splitter(before_table, rcols, target)
                (best_column, gig_value) = columns_sorted[0]
                if gig_value > gig_cutoff:
                    new_path_1 = {'conjunction': conjunct + [(best_column+'_1', build_pred(best_column, 1))],
                                'prediction': None,
                                 'gig_score': gig_value}
                    new_paths.append( new_path_1 ) #true
                    new_path_0 = {'conjunction': conjunct + [(best_column+'_0', build_pred(best_column, 0))],
                                'prediction': None,
                                 'gig_score': gig_value
                                 }
                    new_paths.append( new_path_0 ) #false
                else:
                    #not worth splitting so complete the path with a prediction
                    path['prediction'] = compute_prediction(before_table, target)
                    tree_paths.append(path)
            #end for loop

            current_paths = new_paths
            if current_paths != []:
                n -= 1
            else:
                break  # nothing left to extend so have copied all paths to tree_paths
        #end while loop

        #Generate predictions for all paths that have None
        for path in current_paths:
            conjunct = path['conjunction']
            before_table = generate_table(train, conjunct)
            path['prediction'] = compute_prediction(before_table, target)
            tree_paths.append(path)
        return (tree_paths, oob_list)
    
    #let's build a forest
    forest = []
    for i in range(tree_n):
        (paths, oob) = iterative_build(k)  #always use k for now
        forest.append({'paths': paths, 'weight': None, 'oob': oob})
        
    return forest

#row_index is the row we want to test
#table is 'loan_table'
#k is number of neighbors we take
#columns is 'splitter_columns'
#target is 'Loan_Status'
def euclidean_distance(vector1, vector2):
  return np.sqrt(sum([(a - b) ** 2 for a, b in zip(vector1, vector2)]))

def knn(row_index, table, columns, k, target):
  dist = []
  init_row = table[columns].values[row_index].tolist() #values instead of iloc
  for index, row in table.iterrows():
    if index == row_index:
      continue
    dist.append((index, euclidean_distance(init_row, row[columns].tolist())))
  dist.sort(key=operator.itemgetter(1))
  vote = []
  for (x, y) in dist[:k]:
    vote.append(table[target].values[x]) #same here
  return max(set(vote), key=vote.count)

def knn_tester(table, k, columns, target):
  all_votes = []
  for i in range(len(table)):
    pred = knn(i, table, columns, k, target)
    all_votes.append(pred)
  table['all_votes'] = all_votes
  table['vote_type'] = table.apply(lambda row: predictor_case(row, pred='all_votes', target=target), axis=1)
  p1_types = table['vote_type'].value_counts()
  return accuracy(p1_types)

def oob_test(forest):
  oob_list = []
  for tree in forest:
    oob_list += tree['oob']
  oob_list = list(set(oob_list))
  testing_table = loan_table.loc[loan_table.index.isin(oob_list)]
  testing_table = testing_table.reset_index()
  return testing_table

def oob_vote_taker(row, forest):
    votes = {0:0, 1:0}
    for tree in forest:
        # check to see if it is in oob
        if row['index'] in tree['oob']:
            prediction = tree_predictor(row, tree)
            votes[prediction] += 1
    winner = 1 if votes[1]>votes[0] else 0  #ties go to 0
    return winner

def oob_forest_scores(table, forest, target):
    scratch_table = pd.DataFrame(columns=['prediction', 'actual'])
    scratch_table['prediction'] = table.apply(lambda row: oob_vote_taker(row, forest), axis=1)
    scratch_table['actual'] = table[target]  # just copy the target column
    cases = scratch_table.apply(lambda row: predictor_case(row, pred='prediction', target='actual'), axis=1)
    vc = cases.value_counts()
    return [accuracy(vc), f1(vc), informedness(vc)]