
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn import preprocessing, feature_selection, cross_validation, metrics, cluster


"""
Framework structure
-------------------

Data Cleaning
-------------
Manipulating the data into a form that can be used for learning.
Functions:
	rename_column
	convert_binary_variable
	convert_multiclass_variable
	convert_time_to_seconds
	scale_features
	shuffle_df

Data Visualisation and Clustering
---------------------------------
Plotting and visualising the data to observe trends and using unsupervised learning algorithms
to cluster data together.
Functions:
	kmeans_clustering
	visualise_2d

Feature Selection
-----------------
Using univariate feature selection to discount the features which are correlated least with
the output variable.
Using recursive feature elimination to eliminate the features which have a very small effect
on the value of a given performance metric for a given model.
Functions:
	plot_covariance_matrix
	select_k_best_features
	select_percentile
	compare_two_features
	add_missing_categories
	recursive_feature_selector

Cross-validation
----------------
Deciding which form of cross-validation is needed and employing it to split the data 
as needed. 
Functions:
	is_kfold_cv_needed
	train_cv_test_split
	undersample

Learning
--------
The Learning section contains a range of functions which can be used at any point in the process.
All are employed in functions in the last 3 sections but it is also possible, if not encouraged, 
to use these to dive a little deeper into a model at any stage. Includes functions to 
calculate expectations, plotting roc curves, and getting results for a given set of 
performance metrics for a particular model.
Functions:
	expectation_cost_benefit
	standardise_expectation
	get_scoring_function
	plot_confusion_matrix
	plot_roc_curve
	kfold_plot_roc_curve
	train_test_get_results
	kfold_get_results

Model Selection
---------------
We begin with a wide range of possible models and want to narrow it down to 1 or 2 good models
that we will later evaluate fully. We calculate the results of any chosen performance metrics
and record the results in a dataframe from which we can select the best models.
Recommended use: perform initial fits on 10-15 models by only one or two metrics and narrow
these down to say 5 models. Then evaluate these models more fully using a wider range of
performance metrics.
Functions:
	train_test_record_results_in_df
	kfold_record_results_in_df
	select_k_best_models_from_df
	
Final Model Evaluation
----------------------
Visualising the final model(s) with roc curves, confusion matrices and results from more 
performance metrics.
Functions:
	train_test_display_results
	kfold_display_results
"""




# --------------------------------------------------------------------------------------------




###
# Data cleaning
# Manipulating the data into a form that can be used for learning.
###


def rename_column(df, current_name, new_name):
    """
    Source: self
    
    Renames a column in a dataframe. New column becomes the final column in the DataFrame, to reorder
    use df = df[columns_in_order]
    
    Parameters
    ----------
    df : pandas DataFrame 
    
    current_name : string
        The current name of the column you want to change. Must be present as a column in df.
    
    new_name : string
        The new name to be assigned to that column.
    
    Returns
    -------
    df : pandas DataFrame
        The modified DataFrame.
    
    Examples
    --------
    >>> data = rename_column(data, "column_to_rename", "new_column_name")
    """
    df[new_name] = df[current_name]
    df.drop(current_name, axis=1, inplace=True)
    return df


def convert_binary_variable(df, column_name, positive_category):
    """
    Source: self
    
    Binarises a column in a DataFrame. For a column which can take only two possible values, one value is
    converted to a 1 and the other to a 0.
    
    Parameters
    ----------
    df : pandas DataFrame
    
    column_name : string
        Name of the column in df that you wish to binarise.
    
    positive_category : string (though should work if the values are not strings)
        One of the values that the column can take. The value that you want to assign to be 1.
    
    Returns
    -------
    df : pandas DataFrame
        The modified DataFrame.
    
    Examples
    --------
    >>> data = convert_binary_variable(data, "test_outcome", "pass")
    """
    positives = df[column_name] == positive_category
    negatives = df[column_name] != positive_category
    df.loc[positives, column_name] = 1
    df.loc[negatives, column_name] = 0
    return df


def convert_multiclass_variable(df, column_name, prefix=False):
    """
    Source: Dataquest mission Machine Learning / Machine Learning in Python: Intermediate / 
            Multiclass classification
    
    Your data may contain categorical variables which have more than two categories. In this case, to use
    with some learning algorithms you must convert this column to multiple binary columns. Each column 
    represents one category and contains binary values according to whether that variable is in that category.
    Example:
     Categorical variable | Category 1 | Category 2 | Category 3 
    ----------------------|------------|------------|------------
              1           |     1      |     0      |     0
              2           |     0      |     1      |     0
              3           |     0      |     0      |     1
    
    Parameters
    ----------
    df : pandas DataFrame
    
    column_name : string
        Name of the column you wish to convert. Must be present in df.
    
    prefix : string, default=False
        Prefix for the individual category columns. If not given, then column_name is used as the prefix.
    
    Returns
    -------
    df : pandas DataFrame
        The modified DataFrame. Keeps original column, and adds the columns for each individual category.
    
    Examples
    --------
    >>> data = convert_multiclass_variable(data, "categorical column", prefix="category")
    
    If you wish to delete the original column, combine with:
    >>> data = data.drop("categorical column", axis=1)
    """
    if not prefix:
        prefix = column_name
    dummies = pd.get_dummies(df[column_name], prefix=prefix).astype(int)
    df = pd.concat([df, dummies], axis=1)
    return df


def convert_time_to_seconds(time):
    """
    Source: self
    
    Converts a time in the format hh:mm:ss to the equivalent in total seconds. Can use with DataFrame apply
    method to convert a whole column. 
    (More about apply here: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html)
    
    Parameters
    ----------
    time : string
        The time you wish to convert to total seconds. Must be a string in the format hh:mm:ss.
    
    Returns
    -------
    total_seconds : int
        The equivalent in total seconds of the time parameter.
    
    Examples
    --------
    For individual time:
    >>> seconds = convert_time_to_seconds("hh:mm:ss")
    
    For column:
    >>> df["time_column"] = df["time_column"].apply(convert_time_to_seconds)
    """
    hours = int(time[0:2])
    minutes = int(time[3:5])
    seconds = int(time[6:])
    total_seconds = seconds + 60 * minutes + 3600 * hours
    return total_seconds


def scale_features(df, features):
    """
    Source: Python Data Science Essentials, Chapter 3, Feature creation, page 88
    
    Uses sklearn.preprocessing to scale chosen features in a DataFrame so that a column's mean is 0 and 
    variance is 1. If different features are of different scales then some learning algorithms can be 
    skewed.
    
    Parameters
    ----------
    df : pandas DataFrame
    
    features : 1d array-like object (e.g. numpy array or standard python list)
        List of the features you wish to scale. All the features must be in df.
    
    Returns
    -------
    scaled : 1d numpy array
        Array of the scaled features.
        
    Examples
    --------
    Can reassign to the same columns in the dataframe:
    >>> features = ["feature1", "feature2"]
    >>> df[features] = scale_features(df, features)
    
    Alternatively, can assign to new columns if you wish to preserve the unscaled columns in the dataframe:
    >>> features = ["feature1", "feature2"]
    >>> scaled_features = ["scaled_" + feature for feature in features]
    >>> df[scaled_features] = scale_features(df, features)
    """
    x_scale = df[features]
    scaler = preprocessing.StandardScaler().fit(x_scale)
    scaled = scaler.transform(x_scale)
    return scaled


def shuffle_df(df):
    """
    Source: Dataquest mission  Machine Learning / Machine Learning in Python: Beginner / Cross-validation
    
    Randomly shuffles a dataframe.
    
    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to shuffle.
    
    Returns
    -------
    df : pandas DataFrame
        Shuffled dataframe.
    
    Examples
    --------
    >>> shuffled_df = shuffle_df(df)
    """
    shuffled_index = np.random.permutation(df.index)
    df = df.loc[shuffled_index]
    df = df.reset_index(drop=True)
    return df




# --------------------------------------------------------------------------------------------




###
# Data visualisation and clustering
# Plotting and visualising the data to observe trends and using unsupervised learning algorithms
# to cluster data together.
###


def kmeans_clustering(df, columns, n_clusters, cluster_name="cluster"):
    """
    Source: self
    
    Uses k-means clustering (an unsupervised learning algorithm) to group together data.
    
    Parameters
    ----------
    df : pandas DataFrame
    
    columns : 1d array-like object
        The columns in df by which you want to cluster the data.
    
    n_clusters : int
        Number of clusters to group into. If n_clusters is greater than 2, be sure to use 
        with convert_multiclass_variable before using supervised machine learning.
    
    cluster_name : string, default="cluster"
        Name to give the cluster which will be the new column name on the dataframe.
    
    Returns
    -------
    df : pandas DataFrame
        Modified dataframe with a cluster column containing the cluster that each data point
        belongs to.
    
    kmeans : sklearn.cluster.KMeans object
        The fitted KMeans object. Returned in case further information wants to be obtained from
        the fitting, not just the cluster value for each data point. For more info, see
        http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    
    Examples
    --------
    >>> df, kmeans = kmeans_clustering(df, ["column1", "column2"], n_clusters=3)
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(df[columns])
    df["cluster"] = kmeans.labels_
    return df, kmeans


def visualise_2d(df, x_label, y_label, category_column=False, num_cats=0, xlim=False, ylim=False):
    """
    Source: self
    
    Produces a scatter plot from data in a dataframe. The column names are chosen for the x and y axis, and
    if desired a categorical variable can be represented by making data points a different colour depending
    on which category it is in. For example for a binary classification problem you can plot the two 
    different outcomes as different colours. Or if clustering has been used then a different colour can 
    represent each cluster. Note that categories must be numbered, starting at 0. Currently a maximum of 7 
    colours, but easy to add more: http://matplotlib.org/api/colors_api.html
    
    Parameters
    ----------
    df : pandas DataFrame
    
    x_label : string
        Column name of the variable you want to go on the x-axis. Must be a column in df.
    
    y_label : string
        Column name of the variable you want to go on the y-axis. Must be a column in df.
    
    category_column : string, default=False
        Column name of a categorical variable to be plotted as different colours. Must be a column in df.
    
    num_cats : int, default=0
        Number of categories in the given category_column. Both arguments, num_cats and category_column
        must be given to plot correctly.
    
    xlim : tuple of ints, default=False
        A tuple containing the lower limit and upper limit you want to plot on the x-axis. If not given
        then automatically chosen.
    
    ylim : tuple of ints, default=False
        A tuple containing the lower limit and upper limit you want to plot on the y-axis. If not given 
        then automatically chosen.
    
    Returns
    -------
    None. Only plots graph and displays on screen.
    
    Examples
    --------
    Plotting only x-y, no coloour variation:
    >>> visualise_2d(data, "column1", "column2", xlim=(0, 1), ylim=(-10, 10))
    
    Plotting binary outcome:
    >>> visualise_2d(data, "column1", "column2", "outcome", 2)
    
    Plotting 4 clusters:
    >>> visualise_2d(data, "column1", "column2", "cluster", 4, xlim=(0,1), ylim=(-10,10))
    """
    colours = ['k', 'b', 'r', 'y', 'c', 'g', 'm']
    plt.figure(figsize=(5, 5))
    if category_column and num_cats:
        for n in range(num_cats):
            data_in_cat = df[df[category_column] == n]
            plt.scatter(data_in_cat[x_label], data_in_cat[y_label], color=colours[n], facecolors="none")
    else:
        plt.scatter(df[x_label], df[y_label], facecolors="none")
    
    plt.xlabel(x_label, fontsize=13)
    plt.ylabel(y_label, fontsize=13)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.show()
    pass




# --------------------------------------------------------------------------------------------




###
# Feature selection
# Using univariate feature selection to discount the features which are correlated least with
# the output variable.
# Using recursive feature elimination to eliminate the features which have a very small effect
# on the value of a given performance metric for a given model.
###




def plot_covariance_matrix(df, features):
    """
    Source: Python Data Science Essentials, Chapter 3
                Dimensionality Reduction / Covariance Matrix, page 91
    
    Plots the covariance matrix as a colour map, with different shades representing different levels of 
    correlation between features. If two features are well correlated then you don't need both of them.
    
    Parameters
    ----------
    df : pandas DataFrame
        Dataframe of all the data available.
    
    features : array-like
        The names of the features you want to test how well correlated they are. 
    
    Returns
    -------
    None. Plots graphic to screen.
    
    Examples
    --------
    >>> plot_covariance_matrix(data, ["feature1", "feature2", "feature3"])
    """
    dat = df.loc[:, features].values.astype(float)
    cov_data = np.corrcoef(dat.T)
    img = plt.matshow(cov_data, cmap=plt.cm.winter)
    plt.colorbar(img, ticks=[-1,0,1])
    plt.show()
    pass




###
# Univariate feature selection - used to rule out not very useful features.
# This is more useful when there are more features present as it allows you to eliminate the very weakly 
# correlated features. Works by seeing how well each input feature correlates with the output variable.
# Use recursive feature elimination when you have less features.
# For more info see http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
###


def select_k_best_features(x, y, features, k, test=feature_selection.f_classif):
    """
    Source: self and Python Data Science Essentials, Chapter 3, Feature Selection, page 136
    
    Using univariate feature selection and the sklearn.feature_selection.SelectKBest class, this function
    selects the k best features available. Use with add_missing_categories (below) to ensure all categories 
    of a categorical variable are caught.
    
    Parameters
    ----------
    x : pandas DataFrame
        The data to fit.
    
    y : pandas DataFrame
        The target variable to try to predict. 
    
    features : numpy array
        The names of the features you wish to select from. They must all be columns in x.
    
    k : int
        The number of features to select.
    
    test : sklearn.feature_selection test object, default=feature_selection.f_classif
        The type of test used to test how good a feature is. For other options see
        http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
    
    Returns
    -------
    k_best_features : numpy array
        Array containing the k-best features only.
    
    selector : feature_selection.SelectKBest object
        The selector object used to obtain the best features. Returned in case more than just the best
        features is wanted from the selector. For more info see 
        http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
    
    Examples
    --------
    Select 5 best features:
    >>> best_features, selector = select_k_best_features(x, y, all_features, 5)
    >>> best_features = add_missing_categories(best_features, categorical_features, all_features)
    """
    selector = feature_selection.SelectKBest(test, k=k)
    selector.fit(x, y)
    boolean_mask = selector.get_support()
    k_best_features = features[boolean_mask]
    return k_best_features, selector


def select_percentile(x, y, features, percentile=50, test=feature_selection.f_classif):
    """
    Source: self and Python Data Science Essentials, Chapter 3, Feature Selection, page 136
    
    Using univariate feature selection and the sklearn.feature_selection.SelectPercentile class, this function
    selects the given percentile best features available. Use with add_missing_categories (below) to ensure 
    all categories of a categorical variable are caught.
    
    Parameters
    ----------
    x : pandas DataFrame
        The data to fit.
    
    y : pandas DataFrame
        The target variable to try to predict. 
    
    features : numpy array
        The names of the features you wish to select from. They must all be columns in x.
    
    percentile : int, default=50
        The percentage of features to select.
    
    test : sklearn.feature_selection test object, default=feature_selection.f_classif
        The type of test used to test how good a feature is. For other options see
        http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
    
    Returns
    -------
    best_features : numpy array
        Array containing the best features only.
    
    selector : feature_selection.SelectPercentile object
        The selector object used to obtain the best features. Returned in case more than just the best
        features is wanted from the selector. For more info see 
        http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html
    
    Examples
    --------
    Select 5 best features:
    >>> best_features, selector = select_percentile(x, y, all_features, 5)
    >>> best_features = add_missing_categories(best_features, categorical_features, all_features)
    """
    selector = feature_selection.SelectPercentile(test, percentile=percentile)
    selector.fit(x, y)
    boolean_mask = selector.get_support()
    best_features = features[boolean_mask]
    return best_features, selector

 
def compare_two_features(df, feature_1, feature_2, output, test=feature_selection.f_classif):
    """
    Source: self
    
    Use univariate feature selection to directly compare two features.
    
    Parameters
    ----------
    df : pandas DataFrame
    
    feature_1 : string
        First feature to compare. Must be a column in df.
    
    feature_2 : string
        Second feature to compare. Must be a column in df.
    
    output : string
        The output column name - i.e. the column we want to predict.
    
    test : sklearn.feature_selection test object, default=feature_selection.f_classif
        The type of test used to test how good a feature is. For other options see
        http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
    
    Returns
    -------
    best : string
        The name of the best feature.
    
    worst : string
        The name of the worst feature.
    
    Examples
    --------
    >>> better_feature, poorer_feature = compare_two_features(data, "feature1", "feature2", "outcome")
    """
    features = np.array([feature_1, feature_2])
    Y = df[output]
    X = df[[feature_1, feature_2]]
    selector = feature_selection.SelectKBest(test, k=1)
    selector.fit(X, Y)
    scores_mask = selector.get_support()
    best = features[scores_mask][0]
    not_scores_mask = np.array([not score for score in scores_mask])
    worst = features[not_scores_mask][0]
    return best, worst


def add_missing_categories(selected_features, categorical_features, all_features):
    """
    Source: self
    
    When using select_k_best, select_percentile or other univariate feature selection methods, sometimes
    categories can be missed from a multiclass variable (we use dummy columns for each category). This function
    checks whether all categories for a multiclass feature are present in an array of features and adds 
    them if they are not.
    
    Parameters
    ----------
    selected_features : array
        Features that have been selected by univariate selection.
    
    categorical_features : array
        The name of a categorical feature, i.e. the prefix of the dummy columns of any categorical 
        features (e.g. use stack_depth, not stack_depth_1, stack_depth_2, stack_depth_3 etc...)
    
    all_features : array
        All features available (includes the dummy columns, not the overall variable) - used to check if
        all individual categories are present in selected_features.
    
    Returns
    -------
    selected_features : array
        Modified features, missing categories have been added.
    
    Examples
    --------
    >>> selected_features = ["cluster_0", "cluster_1", "feature"]
    >>> categorical_features = ["cluster", "other_categorical_feature"]
    >>> all_features = ["cluster_0", "cluster_1", "cluster_2", "feature", "other_categorical_feature_0",
                        "other_categorical_feature_1"]
    >>> selected_features = add_missing_categories(selected_features, categorical_features, all_features)
    >>> selected_features
    Out:  ["cluster_0", "cluster_1", "feature", "cluster_2]
    """
    selected_features = list(selected_features)
    for feat in categorical_features:
        # categories for that feature that have been selected
        cats_selected = [cat for cat in selected_features if cat.startswith(feat)]
        # all categories for that feature
        cats_all = [cat for cat in all_features if cat.startswith(feat)]
        if len(cats_selected) != 0 and len(cats_selected) != len(cats_all):
            cats_to_add = [cat for cat in cats_all if cat not in cats_selected]
            for cat in cats_to_add:
                selected_features.append(cat)
    return np.array(selected_features)



###
# Recursive feature elimination - eliminates features until it would have a significant impact on the 
# accuracy (or whichever parameter is chosen)
###


def recursive_feature_selector(model, x, y, scoring="accuracy"):
    """
    Source: self and Python Data Science Essentials, Chapter 3, Feature Selection, page 136
    
    Uses recursive feature selection to select the best features for a given model. Works by eliminating
    features if they do not have a significant impact on the accuracy (or chosen parameter). Features 
    selected depend on the model used so either should only be used when a final model has been 
    chosen (recommended), or it must be done for each individual model before fitting (this can be 
    time consuming so not recommended).
    
    Parameters
    ----------
    model : estimator object
        Estimator object used to evaluate the scoring parameter.
    
    x : pandas DataFrame
        The data to fit. Columns must be all the features that you want to select from.
    
    y : pandas DataFrame
        The target variable to try to predict. 
    
    scoring : string or callable, default="accuracy"
        A string or a scorer callable object/function with signature scorer(estimator, x, y). Examples it
        can take are: "roc_auc", "accuracy", "precision", "recall", "f1". For more info: 
        http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
    
    Returns
    -------
    best_features : array
        The best features for that model.
    
    selector : feature_selection.RFECV object
        The selector object used to obtain the best features. Returned in case more than just the best
        features is wanted from the selector. For more info see 
        http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
    
    Examples
    --------
    >>> best_features, recursive_selector = recursive_feature_selector(LogisticRegression(), x, y)
    """
    features = np.array(x.columns)
    selector = feature_selection.RFECV(estimator=model, cv=3, scoring=scoring)
    selector.fit(x, y)
    boolean_mask = selector.get_support()
    best_features = features[boolean_mask]
    return best_features, selector




# --------------------------------------------------------------------------------------------




###
# Cross-validation
# Deciding which form of cross-validation is needed and employing it to split the data 
# as needed. 
###



def is_kfold_cv_needed(model, x, y, std_threshold=0.01, n_folds=5, random_state=None, scoring="roc_auc"):
    """
    Source: self
    
    Checks if we need k-fold cross-validation for this problem by comparing the size in standard 
    deviation of a given scoring metric across the k folds to a set threshold. In general k-fold 
    is more useful for smaller datasets.
    
    Parameters
    ----------
    model : estimator object
        A model to test with. This will need to be suitable for the problem, e.g. Logistic Regression
        for a binary classification problem, Linear Regression when a continuous variable is the outcome,
        Decision Tree for a multiclassification problem.
    
    x : array-like
        The data to fit. Can be a pandas DataFrame or numpy array, must be at least 2d.
    
    y : array-like
        The target variable to try to predict. 
    
    std_threshold : float, default=0.01
        The threshold for the standard deviation above which we say k-fold is needed.
    
    n_folds : int, default=5
        The number of folds to use.
    
    random_state : int, default=None
        Sets the random state. Used so that results can be reproducible.
    
    scoring : string or callable, default="roc_auc"
        A string or a scorer callable object/function with signature scorer(estimator, x, y). Examples it
        can take are: "roc_auc", "accuracy", "precision", "recall", "f1". For more info: 
        http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
        
        
    Returns
    -------
    A boolean with True if the standard deviation is greater than the threshold, i.e. k-fold is needed, or 
    False if k-fold is not needed.
    
    Examples
    --------
    >>> if is_kfold_cv_needed(LogisticRegression(), x, y, random_state=7, std_threshold=0.1):
            (use kfold...)
    """
    kf = cross_validation.KFold(len(x), n_folds=n_folds, shuffle=True, random_state=random_state)
    scores = cross_validation.cross_val_score(model, x, y, scoring="roc_auc", cv=kf)
    s = scores.std()
    
    return s > std_threshold
    

def train_cv_test_split(x, y, split_sizes=[0.6, 0.2, 0.2], random_state=None):
    """
    Source: self
    
    Split the data set into a training set, cross-validation sets (number is inputted), and a test set.
    
    Parameters
    ----------
    x : pandas DataFrame
        The data to fit. 
    
    y : pandas DataFrame
        The target variable to try to predict. 
    
    split_sizes : list of floats, default=[0.6, 0.2, 0.2]
        The size of each split. In order: [train_size, cv_size_1, cv_size_2, ... , test_size]
    
    random_state : int, default=None
        Decides random state used. Used to make results repeatable.
    
    Returns
    -------
    X : list of DataFrames
        List of the dataframes in the order train, cvs, test for the data being used to fit.
    
    Y : list of DataFrames
        List of the dataframes in the order train, cvs, test for the target variable.
    
    Examples
    --------
    >>> X, Y = train_cv_test_split(x, y, split_sizes=[0.5, 0.3, 0.2], random_state=7)
    >>> x_train, x_cv, x_test = X
    >>> y_train, y_cv, y_test = Y
    """
    if sum(split_sizes) != 1.0:
        raise ValueError("The sum of the split sizes must add up to 1.")
    train_size = split_sizes.pop(0)
    x_train, x_cv_test, y_train, y_cv_test = cross_validation.train_test_split(x, y, test_size=sum(split_sizes), 
                                                                               random_state=random_state)
    X = []
    Y = []
    X.append(x_train)
    Y.append(y_train)
    
    number_of_cvs = len(split_sizes) - 1
    for i in range(number_of_cvs):
        test_size_i = sum(split_sizes[i+1:]) / sum(split_sizes[i:])
        x_cv_i, x_cv_test, y_cv_i, y_cv_test = cross_validation.train_test_split(x_cv_test, y_cv_test, 
                                                                                 test_size=test_size_i)
        X.append(x_cv_i)
        Y.append(y_cv_i)
    X.append(x_cv_test)
    Y.append(y_cv_test)
    return X, Y


def undersample(x_train, y_train, positive_fraction=0.5, random_state=None):
    """
    Source: self, with some from Dataquest (how to randomise dataframe)
    
    Delete positive instances from the training set so that the two outcome are balanced (or in 
    a set proportion).
    
    Parameters
    ----------
    x_train : pandas DataFrame
        The training data to fit. 
    
    y_train : pandas DataFrame
        The training target variable to try to predict. 
    
    positive_fraction : float, default=0.5
        Fraction of the positive outcome to be present in the undersampled data set.
    
    random_state : int, default=None
        Used so that results are repeatable.
    
    Returns
    -------
    x_train_under : pandas DataFrame
        The training data undersampled to the set percentage.
    
    y_train_under : pandas DataFrame
        The training target variable undersampled to the set percentage
    
    Examples
    --------
    >>> x_train_under50, y_train_under50 = undersample(x_train, y_train, 0.6, random_state=7)
    """
    # separate training set into positive and negative instances
    positive_mask = (y_train == 1)
    negative_mask = (y_train == 0)
    x_train_positive = x_train[positive_mask]
    x_train_negative = x_train[negative_mask]
    y_train_positive = y_train[positive_mask]
    y_train_negative = y_train[negative_mask]
    num_positive = len(y_train_positive)
    num_negative = len(y_train_negative)
    
    # shuffle order of positive instances
    shuffled_index_positive = np.random.permutation(x_train_positive.index)
    x_shuffled_positive = x_train_positive.loc[shuffled_index_positive]
    y_shuffled_positive = y_train_positive.loc[shuffled_index_positive]
    x_train_positive = x_shuffled_positive.reset_index(drop=True)
    y_train_positive = y_shuffled_positive.reset_index(drop=True)
    
    # delete some positive instances
    new_num_positive = int(num_negative * (positive_fraction / (1 - positive_fraction)))
    x_train_positive = x_train_positive.iloc[:new_num_positive]
    y_train_positive = y_train_positive.iloc[:new_num_positive]
    
    # join positive and negative instances together again and randomise order
    x_train_under = pd.concat([x_train_positive, x_train_negative], ignore_index=True)
    y_train_under = pd.concat([y_train_positive, y_train_negative], ignore_index=True)
    shuffled_index = np.random.permutation(y_train_under.index)
    x_train_under = x_train_under.loc[shuffled_index]
    y_train_under = y_train_under.loc[shuffled_index]
    x_train_under = x_train_under.reset_index(drop=True)
    y_train_under = y_train_under.reset_index(drop=True)
    
    return x_train_under, y_train_under




# --------------------------------------------------------------------------------------------




###
# Learning
# The Learning section contains a range of functions which can be used at any point in the process.
# All are employed in functions in the last 3 sections but it is also possible, if not encouraged, 
# to use these to dive a little deeper into a model at any stage. Includes functions to 
# calculate expectations, plotting roc curves, and getting results for a given set of 
# performance metrics for a particular model.
###


def expectation_cost_benefit(y_true, y_pred, cb_matrix):
    """
    Source:
    http://nbviewer.jupyter.org/github/podopie/DAT18NYC/blob/master/classes/13-expected_value_cost_benefit_analysis.ipynb
    
    Combines the cost benefit matrix with the confusion matrix to get an expectation value. Use 
    standardise_expectation to convert this to a value between 0 and 1.
    
    Parameters
    ----------
    y_true : pandas DataFrame
        Ground truth (correct) target values.
    
    y_pred : pandas DataFrame
        Estimated targets as returned by a classifier.
    
    cb_matrix : 2d array
        Cost benefit matrix.
    
    Returns
    -------
    expectation : float
        The expectation value.
    
    Examples
    --------
    >>> exp = expectation_cost_benefit(y_true, y_pred, cost_benefit_matrix)
    """
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    probabilities = confusion_matrix / np.sum(confusion_matrix)
    expectation = np.sum(probabilities * cb_matrix)
    return expectation


def standardise_expectation(y_true, y_pred, cb_matrix, min_expectation, max_expectation):
    """
    Source: self
    
    Standardises expectation value to be between 0 and 1 using the formula
        (expectation - min_expectation) / (max_expectation - min_expectation).
    The maximum and minimum expectation values should be defined by the user as they will depend 
    on the problem at hand. The maximum will likely always be for perfect predictions so you 
    can calculate it using
    	max_expectation = expectation_cost_benefit(y_true, y_true, cb_matrix)
    The minimum will depend on the problem. For example, if you have a skewed dataset where most 
    of the true outcomes are positive then you may want to set your minimum as the expectation 
    value if all outcomes are predicted to be positive. You would do this like:
    	y_all_positive = np.ones_like(y_true)
    	min_expectation = expectation_cost_benefit(y_true, y_all_positive, cb_matrix)
    You could also define it as when you get all predictions wrong. It depends on the problem
    at hand.
    
    Parameters
    ----------
    y_true : pandas DataFrame
        Ground truth (correct) target values.
    
    y_pred : pandas DataFrame
        Estimated targets as returned by a classifier.
    
    cb_matrix : 2d array
        Cost benefit matrix.
    
    min_expectation : int
        Minimum expectation value. How this is calculated can depend on the problem.
    
    max_expectation : int
        Maximum expectation value. This will be if the model predicts perfectly.
    
    Returns
    -------
    stand_exp : float
        Standardised expectation value.
    
    Examples
    --------
    >>> stand_exp = standardise_expectation(y_true, y_pred, cost_benefit_matrix, min_expectation, max_expectation)
    """
    expectation = expectation_cost_benefit(y_true, y_pred, cb_matrix)
    stand_exp = (expectation - min_expectation) / (max_expectation - min_expectation)
    return stand_exp


def get_scoring_function(scoring, cb_matrix=np.array([]), min_expectation=False, max_expectation=False):
    """
    Source: self
    
    Makes or gets a scorer from a performance metric function. Uses metrics.make_scorer to make a callable
    scoring function with the signature scorer(y_true, y_pred) for user defined functions. Also uses
    metrics.get_scorer to get the scoring function for a built-in function. For more info on both, see
    http://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
    This function is used later in many results calculating functions.
    
    Parameters
    ----------
    scoring : string or callable, default="roc_auc"
        A string or a scorer callable object/function with signature scorer(estimator, x, y). Examples it
        can take are: "roc_auc", "accuracy", "precision", "recall", "f1". For more info: 
        http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values 
        In addition to the standard predefined values, it can also take "sensitivity", "specificity",
        "expectation" (from cost benefit matrix) and "stand_exp" (standardised version of expectation).
    
    cb_matrix : array, default=np.array([])
        The cost benefit matrix for this problem. This is argument is not needed unless scoring 
        is either "expectation" or "stand_exp"
    
    min_expectation : float, default=False
        The minimum expectation value for this problem. Only needs to be specified if scoring="stand_exp".
    
    max_expectation : float, default=False
        The maximum expectation value for this problem. Only needs to be specified if scoring="stand_exp".
    
    Returns
    -------
    scorer : callable
        A callable function that must be called with the signature scorer(model, x_test, y_test)
    
    Examples
    --------
    >>> scorer = get_scoring_function("expectation", cb_matrix=np.array([[0, 12], [0, 18]]))
    >>> expectation = scorer(LogisticRegression(), x_test, y_test)
    """
    if scoring == "sensitivity":
        sensitivity_scorer = metrics.make_scorer(metrics.recall_score, pos_label=1)
        scorer = metrics.get_scorer(sensitivity_scorer)
    elif scoring == "specificity":
        specificity_scorer = metrics.make_scorer(metrics.recall_score, pos_label=0)
        scorer = metrics.get_scorer(specificity_scorer)
    elif scoring == "expectation":
        if len(cb_matrix) == 0:
            raise ValueError("If scoring is 'expectation' then cb_matrix must be given")
        expectation_scorer = metrics.make_scorer(expectation_cost_benefit, cb_matrix=cb_matrix)
        scorer = metrics.get_scorer(expectation_scorer)
    elif scoring == "stand_exp":
        if (len(cb_matrix) == 0) or (not min_expectation) or (not max_expectation):
            raise ValueError("If scoring is 'stand_exp' then cb_matrix, min_expectation and max_expectation must"
                             " all be given")
        stand_exp_scorer = metrics.make_scorer(standardise_expectation, cb_matrix=cb_matrix,
                                               min_expectation=min_expectation, max_expectation=max_expectation)
        scorer = metrics.get_scorer(stand_exp_scorer)
    else:
        scorer = metrics.get_scorer(scoring)
    
    return scorer


def plot_confusion_matrix(y_true, y_pred):
    """
    Source: Python Data Science Essentials, Chapter 3 
            Dimensionality reduction / Covariance matrix, page 91 
          & Scoring functions / Multilabel classification, page 115
    
    Calculates confusion matrix and prints it to the screen. Also plots it as a colour map image.
    
    Parameters
    ----------
    y_true : pandas DataFrame
        Ground truth (correct) target values.
    
    y_pred : pandas DataFrame
        Estimated targets as returned by a classifier.
    
    Returns
    -------
    cm : 2d array
        Confusion matrix.
    
    Examples
    --------
    >>>
    """
    cm = metrics.confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)
    img = plt.matshow(cm, cmap=plt.cm.winter)
    plt.colorbar(img)
    plt.ylabel("Actual outcomes")
    plt.xlabel("Predicted outcomes")
    plt.show()
    return cm


def plot_roc_curve(fpr, tpr):
    """
    Source: self and Dataquest, Machine Learning / Machine Learning in Python: Beginner / Cross-validation
    
    Plot Reciver Operator Characteristics curve with false positive rate on x axis and true positive 
    rate on y axis. Default discrimination threshold for positive/negative outcomes is a probability of 50%.
    We can vary the threshold to find the best. TPR is proportion predicted positive that were positive.
    FPR is proportion predicted positive that were negative. We want low FPR, high TPR.
    
    Parameters
    ----------
    fpr : array
        False positive rates as calculated using metrics.roc_curve.
    
    tpr : array
        True positive rates as calculated using metrics.roc_curve.
        
    Returns
    -------
    None. Plots to screen only.
    
    Examples
    --------
    >>> fpr, tpr, thresholds = metrics.roc_curve(y_true, pass_probability)
    >>> plot_roc_curve(fpr, tpr)
    """
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color="blue")
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()
    pass


def kfold_plot_roc_curve(model, x, y, kf):
    """
    Source:
    http://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    
    Plots ROC curve for k-fold cross-validation. Plots the curve for each fold, and the average of all folds.
    Also plots the variance at each point in the curve as filled grey areas. See first source link for 
    examples of what the plots look like.
    
    Parameters
    ----------
    model : estimator object
        Model used to obtain roc_curve with.
    
    x : pandas DataFrame
        The training data to fit. 
    
    y : pandas DataFrame
        The training target variable to try to predict. 
    
    kf : cross_validation.KFold object
        The KFold object used to obtain the folds.
    
    Returns
    -------
    None, only prints to the screen.
    
    Examples
    --------
    >>> kfold_plot_roc_curve(LogisticRegression(), x, y, kfold)
    """
    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    plt.figure(figsize=(5, 5))

    for i, (train, test) in enumerate(kf):
        model.fit(x.loc[train], y.loc[train])
        pass_probabilities = model.predict_proba(x.loc[test])[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y.loc[test], pass_probabilities)

        plt.plot(fpr, tpr, 'b', alpha=0.15)
        tpr = scipy.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, 'b')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()
    pass


def train_test_get_results(model, x_train, x_test, y_train, y_test, scoring_metrics, cb_matrix=np.array([]),
                           min_expectation=False, max_expectation=False):
    """
    Source: self
    
    Gets results of key metrics for a model using holdout cross-validation. The metrics can be specified, 
    and the function makes use of get_scoring_function to get the different functions for each metric. 
    
    Parameters
    ----------
    model : estimator object
        Model used to obtain results with.
        
    x_train : pandas DataFrame
        The training data to fit. 
    
    x_test : pandas DataFrame
        The test data to predict from.
    
    y_train : pandas DataFrame
        The training target variable to try to predict. 
    
    y_test : pandas DataFrame
        The test target variable to try to predict.
    
    scoring_metrics : list
        Each value in the list must be a string or a scorer callable object/function with signature
        scorer(estimator, x, y). Examples it can take are: "roc_auc", "accuracy", "precision",
        "recall", "f1". For more info: 
            http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values 
        In addition to the standard predefined values, it can also take "sensitivity", "specificity",
        "expectation" (from cost benefit matrix) and "stand_exp" (standardised version of expectation).
    
    cb_matrix : array, default=np.array([])
        The cost benefit matrix for this problem. This argument is not needed unless scoring 
        is either "expectation" or "stand_exp"
    
    min_expectation : float, default=False
        The minimum expectation value for this problem. Only needs to be specified if scoring="stand_exp".
    
    max_expectation : float, default=False
        The maximum expectation value for this problem. Only needs to be specified if scoring="stand_exp".
    
    Returns
    -------
    results : dict
        Dictionary object with the scoring_metrics as the keys, and their estimated values as the values.
    
    Examples
    --------
    >>> scoring_metrics = ["accuracy", "roc_auc", "expectation"]
    >>> results = train_test_get_results(model, x_train, x_test, y_train, y_test, scoring_metrics,
                                         cost_benefit_matrix)
    """
    model.fit(x_train, y_train)
    results = {}
    
    for scoring in scoring_metrics:
        scorer = get_scoring_function(scoring, cb_matrix, min_expectation, max_expectation)
        results[scoring] = scorer(model, x_test, y_test)
    
    return results


def kfold_get_results(model, x, y, kf, scoring_metrics, cb_matrix=np.array([]), min_expectation=False,
                      max_expectation=False):
    """
    Source: self
    
    Gets results of key metrics for a model using k-fold cross-validation. The metrics can be specified, 
    and the function makes use of get_scoring_function to get the different functions for each metric.  
    
    Parameters
    ----------
    model : estimator object
        Model used to obtain results with.
    
    x : pandas DataFrame
        The training data to fit. 
    
    y : pandas DataFrame
        The training target variable to try to predict. 
    
    kf : cross_validation.KFold object
        The KFold object used to obtain the folds.
    
    scoring_metrics : list
        Each value in the list must be a string or a scorer callable object/function with signature
        scorer(estimator, x, y). Examples it can take are: "roc_auc", "accuracy", "precision",
        "recall", "f1". For more info: 
            http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values 
        In addition to the standard predefined values, it can also take "sensitivity", "specificity",
        "expectation" (from cost benefit matrix) and "stand_exp" (standardised version of expectation).
    
    cb_matrix : array, default=np.array([])
        The cost benefit matrix for this problem. This argument is not needed unless scoring 
        is either "expectation" or "stand_exp"
    
    min_expectation : float, default=False
        The minimum expectation value for this problem. Only needs to be specified if scoring="stand_exp".
    
    max_expectation : float, default=False
        The maximum expectation value for this problem. Only needs to be specified if scoring="stand_exp".
    
    Returns
    -------
    results : dict
        Dictionary object with the scoring_metrics as the keys, and their estimated values as the values.
    
    Examples
    --------
    >>> scoring_metrics = ["accuracy", "roc_auc", "expectation"]
    >>> results = kfold_get_results(model, x, y, kfold, scoring_metrics, cost_benefit_matrix)
    """
    scores = {scoring: [] for scoring in scoring_metrics}
    
    for train_index, test_index in kf:
        # split x and y into training and test set for this fold
        x_train_kf, x_test_kf = x.loc[train_index], x.loc[test_index]
        y_train_kf, y_test_kf = y.loc[train_index], y.loc[test_index]
        
        fold_results = train_test_get_results(model, x_train_kf, x_test_kf, y_train_kf, y_test_kf, scoring_metrics,
                                              cb_matrix, min_expectation, max_expectation)
        
        for scoring in scoring_metrics:
            scores[scoring].append(fold_results[scoring])
    
    results = {scoring: np.mean(scores[scoring]) for scoring in scoring_metrics}
        
    return results




# --------------------------------------------------------------------------------------------




###
# Model Selection
# We begin with a wide range of possible models and want to narrow it down to 1 or 2 good models
# that we will later evaluate fully. We calculate the results of any chosen performance metrics
# and record the results in a dataframe from which we can select the best models.
# Recommended use: perform initial fits on 10-15 models by only one or two metrics and narrow
# these down to say 5 models. Then evaluate these models more fully using a wider range of
# performance metrics.
###



def train_test_record_results_in_df(models_df, x_train, x_test, y_train, y_test, scoring_metrics,
                                    cb_matrix=np.array([]), min_expectation=False, max_expectation=False):
    """
    Source : self
    
    Using holdout cross-validation, performance metrics are calculated for a set of models and are
    recorded in a dataframe.
    
    Parameters
    ----------
    models_df : pandas DataFrame
    	A dataframe of the models we want to evaluate. Each row represents a model and must have
    	the columns "Model Name", where the name or label for that model is stored, and "Model",
    	where the actual model is stored.
    
    x_train : pandas DataFrame
        The training data to fit. 
    
    x_test : pandas DataFrame
        The test data to predict from.
    
    y_train : pandas DataFrame
        The training target variable to try to predict. 
    
    y_test : pandas DataFrame
        The test target variable to try to predict.
    
    scoring_metrics : list
        Each value in the list must be a string or a scorer callable object/function with signature
        scorer(estimator, x, y). Examples it can take are: "roc_auc", "accuracy", "precision",
        "recall", "f1". For more info: 
            http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values 
        In addition to the standard predefined values, it can also take "sensitivity", "specificity",
        "expectation" (from cost benefit matrix) and "stand_exp" (standardised version of expectation).
    
    cb_matrix : array, default=np.array([])
        The cost benefit matrix for this problem. This is argument is not needed unless scoring 
        is either "expectation" or "stand_exp"
    
    min_expectation : float, default=False
        The minimum expectation value for this problem. Only needs to be specified if scoring="stand_exp".
    
    max_expectation : float, default=False
        The maximum expectation value for this problem. Only needs to be specified if scoring="stand_exp".
    
    Returns 
    -------
    results_df : pandas DataFrame
    	Modified dataframe which now contains columns for each of the chosen scoring metrics for each 
    	model.
    
    Examples
    --------
    >>> results = train_test_record_results_in_df(models, x_train, x_test, y_train, y_test,
    											  scoring_metrics, cost_benefit_matrix, 
    											  min_expectation, max_expectation)
    """
    results_df = pd.DataFrame()
    
    for i, row in models_df.iterrows():
        print("Fitting {}...".format(row["Model Name"]))
        model = row["Model"]
        model_results = train_test_get_results(model, x_train, x_test, y_train, y_test, scoring_metrics, cb_matrix,
                                               min_expectation, max_expectation)
        
        # Add the model_results to the row
        for scoring in scoring_metrics:
            row[scoring] = model_results[scoring]
        
        results_df = results_df.append(row)
    
    return results_df


def kfold_record_results_in_df(models_df, x, y, kf, scoring_metrics, cb_matrix=np.array([]),
							   min_expectation=False, max_expectation=False):
    """
    Source : self
    
    Using k-fold cross-validation, performance metrics are calculated for a set of models and are
    recorded in a dataframe.
    
    Parameters
    ----------
    models_df : pandas DataFrame
    	A dataframe of the models we want to evaluate. Each row represents a model and must have
    	the columns "Model Name", where the name or label for that model is stored, and "Model",
    	where the actual model is stored.
    
    x : pandas DataFrame
        The training data to fit. 
    
    y : pandas DataFrame
        The training target variable to try to predict. 
    
    kf : cross_validation.KFold object
        KFold object which contains the indices for each of the folds.
        
    scoring_metrics : list
        Each value in the list must be a string or a scorer callable object/function with signature
        scorer(estimator, x, y). Examples it can take are: "roc_auc", "accuracy", "precision",
        "recall", "f1". For more info: 
            http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values 
        In addition to the standard predefined values, it can also take "sensitivity", "specificity",
        "expectation" (from cost benefit matrix) and "stand_exp" (standardised version of expectation).
    
    cb_matrix : array, default=np.array([])
        The cost benefit matrix for this problem. This is argument is not needed unless scoring 
        is either "expectation" or "stand_exp"
    
    min_expectation : float, default=False
        The minimum expectation value for this problem. Only needs to be specified if scoring="stand_exp".
    
    max_expectation : float, default=False
        The maximum expectation value for this problem. Only needs to be specified if scoring="stand_exp".
    
    Returns 
    -------
    results_df : pandas DataFrame
    	Modified dataframe which now contains columns for each of the chosen scoring metrics for each 
    	model.
    
    Examples
    --------
    >>> results = kfold_record_results_in_df(models, x, y, kfold, scoring_metrics, cost_benefit_matrix, 
    										 min_expectation, max_expectation)
    """
    results_df = pd.DataFrame()
    
    for i, row in models_df.iterrows():
        print("Fitting {}...".format(row["Model Name"]))
        model = row["Model"]
        model_results = kfold_get_results(model, x, y, kf, scoring_metrics, cb_matrix,
                                          min_expectation, max_expectation)
        
        # Add the model_results to the row
        for scoring in scoring_metrics:
            row[scoring] = model_results[scoring]
        
        results_df = results_df.append(row)
    
    return results_df


def select_k_best_models_from_df(results_df, k, scoring_parameter, higher_is_better=True):

    """
    Source: self
    
    From a dataframe containing the results of many different models for many different metrics, the 
    best k models according to a chosen metric are chosen. 
    
    Parameters
    ----------
    results_df : pandas DataFrame
        Dataframe of results with each model as a row and each metric as a column.
    
    k : int
        Number of models to select.
    
    scoring_parameter : string
        Chosen parameter to evaluate by. Must be a column in results_df.
    
    higher_is_better : bool, default=True
        True if for the chosen scoring_parameter, a higher value is better.
    
    Returns
    -------
    best_models : pandas DataFrame
        Dataframe of the chosen best_models
    
    Examples
    --------
    >>> best_models = select_k_best_models_from_df(results, 1, "roc_auc")
    """
    
    sorted_results = results_df.sort_values(scoring_parameter, ascending=(not higher_is_better))
    best_models = sorted_results.iloc[:k]
    return best_models




# --------------------------------------------------------------------------------------------




###
# Final model evaluation
# Visualising the final model(s) with roc curves, confusion matrices and results from more 
# performance metrics.
###


def train_test_display_results(model, x_train, x_test, y_train, y_test, scoring_metrics, cb_matrix=np.array([]),
                               min_expectation=False, max_expectation=False):
    """
    Source: self
    
    Displaying results for a model to the screen using holdout validation. Uses functions from above:
    	plot_confusion_matrix
    	plot_roc_curve
    	train_test_get_results
    
    Parameters
    ----------
    model : estimator object
        Model used to obtain roc_curve with.
        
    x_train : pandas DataFrame
        The training data to fit. 
    
    x_test : pandas DataFrame
        The test data to predict from.
    
    y_train : pandas DataFrame
        The training target variable to try to predict. 
    
    y_test : pandas DataFrame
        The test target variable to try to predict.
    
    scoring_metrics : list
        Each value in the list must be a string or a scorer callable object/function with signature
        scorer(estimator, x, y). Examples it can take are: "roc_auc", "accuracy", "precision",
        "recall", "f1". For more info: 
            http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values 
        In addition to the standard predefined values, it can also take "sensitivity", "specificity",
        "expectation" (from cost benefit matrix) and "stand_exp" (standardised version of expectation).
    
    cb_matrix : array, default=np.array([])
        The cost benefit matrix for this problem. This argument is not needed unless scoring 
        is either "expectation" or "stand_exp"
    
    min_expectation : float, default=False
        The minimum expectation value for this problem. Only needs to be specified if scoring="stand_exp".
    
    max_expectation : float, default=False
        The maximum expectation value for this problem. Only needs to be specified if scoring="stand_exp".
    
    Returns
    -------
    results : dict
        Dictionary object with the scoring_metrics as the keys, and their estimated values as the values.
    
    Examples
    --------
    >>> scoring_metrics = ["accuracy", "roc_auc", "expectation"]
    >>> results = train_test_display_results(model, x_train, x_test, y_train, y_test, scoring_metrics,
                                         cost_benefit_matrix)
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    pass_probability = model.predict_proba(x_test)[:, 1]
    
    cm = plot_confusion_matrix(y_test, y_pred)
    
    ###
    # plotting roc curves and calculating auc score
    # Default discrimination threshold for positive/negative outcomes is a probability of 50%. We can vary the 
    # threshold to find the best. 
    # true positive rate - tpr is proportion predicted positive that were positive.
    # false positive rate - fpr is proportion predicted positive that were negative.
    # We want low fpr, high tpr
    ###
    
    print("\nROC curve:\n")
    
    # roc curve (receiver operator curve) calculates tpr and fpr at each discrimination threshold
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pass_probability)
    plot_roc_curve(fpr, tpr)
    
    
    print("\nClassification report:\n")
    print(metrics.classification_report(y_test, y_pred))
    
    print("Calculating performance metrics...")
    results = train_test_get_results(model, x_train, x_test, y_train, y_test, scoring_metrics, cb_matrix,
                                     min_expectation, max_expectation)
    
    for scoring in scoring_metrics:
        print("%s: %.5f" % (scoring, results[scoring]))
    
    return results


def kfold_display_results(model, x, y, kf, scoring_metrics, cb_matrix=np.array([]), min_expectation=False,
                          max_expectation=False):
    """
    Source: self (and see sources for kfold_plot_roc_curve
    
    Displaying results for a model to the screen using k-fold cross-validation. Uses same code from 
    the following functions from above:
    	kfold_plot_roc_curve
    	kfold_get_results
    Uses the same code rather than just using the functions as it saves fitting each model more than 
    once for each fold. 
    
    Parameters
    ----------
    model : estimator object
        Model used to obtain results with.
    
    x : pandas DataFrame
        The training data to fit. 
    
    y : pandas DataFrame
        The training target variable to try to predict. 
    
    kf : cross_validation.KFold object
        The KFold object used to obtain the folds.
    
    scoring_metrics : list
        Each value in the list must be a string or a scorer callable object/function with signature
        scorer(estimator, x, y). Examples it can take are: "roc_auc", "accuracy", "precision",
        "recall", "f1". For more info: 
            http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values 
        In addition to the standard predefined values, it can also take "sensitivity", "specificity",
        "expectation" (from cost benefit matrix) and "stand_exp" (standardised version of expectation).
    
    cb_matrix : array, default=np.array([])
        The cost benefit matrix for this problem. This argument is not needed unless scoring 
        is either "expectation" or "stand_exp"
    
    min_expectation : float, default=False
        The minimum expectation value for this problem. Only needs to be specified if scoring="stand_exp".
    
    max_expectation : float, default=False
        The maximum expectation value for this problem. Only needs to be specified if scoring="stand_exp".
    
    Returns
    -------
    results : dict
        Dictionary object with the scoring_metrics as the keys, and their estimated values as the values.
    
    Examples
    --------
    >>> scoring_metrics = ["accuracy", "roc_auc", "expectation"]
    >>> results = kfold_display_results(model, x, y, kfold, scoring_metrics, cost_benefit_matrix)
    """
    scores = {scoring: [] for scoring in scoring_metrics}
    
    # for roc curves
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    plt.figure(figsize=(5, 5))
    
    # loop for k-fold iterator. train_index and test_index are index masks for each fold
    for i, (train_index, test_index) in enumerate(kf):
        print("Computing fold {}...".format(i))
        
        # split x and y into training and test set for this fold
        x_train_kf, x_test_kf = x.loc[train_index], x.loc[test_index]
        y_train_kf, y_test_kf = y.loc[train_index], y.loc[test_index]
        
        
        fold_results = train_test_get_results(model, x_train_kf, x_test_kf, y_train_kf, y_test_kf, scoring_metrics,
                                              cb_matrix, min_expectation, max_expectation)
        
        for scoring in scoring_metrics:
            scores[scoring].append(fold_results[scoring])
        
        
        # fit with training set and predict with test set
        model.fit(x_train_kf, y_train_kf)
        pass_probability = model.predict_proba(x_test_kf)[:,1]
        
        
        # plotting roc curve for this fold
        fpr, tpr, thresholds = metrics.roc_curve(y_test_kf, pass_probability)
        plt.plot(fpr, tpr, 'b', alpha=0.15)
        tpr = scipy.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
    
    print("\nROC curve:\n")
    
    # averaging and plotting roc curve
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, 'b')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()
    
    
    results = {scoring: np.mean(scores[scoring]) for scoring in scoring_metrics}
    
    for scoring in scoring_metrics:
        print("%s: %.5f" % (scoring, results[scoring]))
    
    
    return results
