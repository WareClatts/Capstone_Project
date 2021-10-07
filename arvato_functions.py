

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##############################################################################################################
#################################### DATA PREPARATION ########################################################
##############################################################################################################

def encode_cols(df, cols):

    df_dummies = pd.get_dummies(df[cols]) #,drop_first=True
    df_ = df.drop(labels=cols, axis=1)
    df_[df_dummies.columns] = df_dummies.iloc[:,:]

    return df_

##############################################################################################################

def prepare_cat_cols(cleaned_df, continuous_vars, to_dummy):
    
    """Columns to categorical type and one-hot encoding"""
    
    ## Columns with strings
    stringcols=[]
    for col in cleaned_df.columns:
        if cleaned_df.dtypes[col] == np.object:
            print(f"Data type of column {col} is object")
            stringcols.append(col)
            
    ## Cateogrical columns
    for col in cleaned_df.columns:
        if col in continuous_vars:
            pass
        else:
            cleaned_df[col] = cleaned_df[col].astype('str')
            cleaned_df[col] = cleaned_df[col].astype('category')
            
    ## One-hot encoding of categorical variables 
    # Will result in many columns but may be needed as string type cannot go into PCA
    # Can do with string-type variables only for now, or all
    
    catcols = []
    for col in cleaned_df.columns:
        if col in continuous_vars:
            pass
        else:
            catcols.append(col)

    if to_dummy == 'string_cols':
        prep_df = encode_cols(cleaned_df, stringcols)
    elif to_dummy == 'all_cat_cols':
        prep_df = encode_cols(cleaned_df, catcols)
    elif to_dummy == 'None':
        prep_df=cleaned_df
    else:
        if len(stringcols) > 0:
            prep_df = encode_cols(cleaned_df, to_dummy+stringcols)
        else:
            prep_df = encode_cols(cleaned_df, to_dummy)
    
    ## index
    try:
        prep_df.set_index('LNR', inplace=True)
    except:
        pass

    return prep_df

##############################################################################################################

def prepare_famd(df, continuous_vars, to_drop, reduce_columns, subsample=0.5):
    
    """Preparing data for FAMD"""
    test = df.sample(frac=subsample, replace=False, random_state=1).set_index('LNR')
    
    if reduce_columns == True:
        test.drop(labels=to_drop, axis=1, inplace=True)
    
    test = prepare_cat_cols(test, continuous_vars, 'None')
    
    print(f'shape of prepared data = {test.shape[0]} rows, {test.shape[1]} columns')
        
    return test

##############################################################################################################

def prepare_mca(df, to_drop, continuous_vars, reduce_columns, subsample=0.5):
    
    """Preparing data for MCA"""
    
    test = df.sample(frac=subsample, replace=False, random_state=1).set_index('LNR')
    
    test.drop(labels=continuous_vars, axis=1, inplace=True)
    
    if reduce_columns == True:
        test.drop(labels=to_drop, axis=1, inplace=True)
    
    test = prepare_cat_cols(test, continuous_vars, 'all_cat_cols') #'None'
    
    test = test.replace([-1],100)
    
    print(f'shape of MCA prepared data = {test.shape[0]} rows, {test.shape[1]} columns')
        
    return test

##############################################################################################################

def prepare_pca(df, to_drop, continuous_vars, subsample=0.5):
    
    """Preparing data for MCA"""
    
    test = df.sample(frac=subsample, replace=False, random_state=1).set_index('LNR')
    
    test = test[continuous_vars]
    
    print(f'shape of PCA prepared data = {test.shape[0]} rows, {test.shape[1]} columns')
        
    return test

##############################################################################################################
######################################### CHECKS #############################################################
##############################################################################################################

def datatypes_summary(df):
    column_types = []
    data_types = []
    for col in df.columns:
        column_types.append(df[col].dtype)
        data_types.append(type(df[col].iloc[0]))
    summary = pd.DataFrame(index=df.columns)
    summary['column_types'] = [str(x) for x in column_types]
    summary['data_types'] = [str(x) for x in data_types]
    return summary.groupby(['column_types', 'data_types']).count()

##############################################################################################################

def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

##############################################################################################################

def check_cols_match(df1, df2, df1name, df2name):
    
    """Check that columns in one dataframe are in another - should be same given same prep"""
    
    set1=set(df1.columns)
    set2=set(df2.columns)
    
    df1_to_drop = []
    df2_to_drop = []
    
    diff = set1.difference(set2)
    if len(diff) > 0:
        print(f'in {df1name} but not {df2name}:')
        print(f'{diff}')
        df1_to_drop = list(diff)
    else:
        print(f'no columns in {df1name} that are not in {df2name}')
    diff = set2.difference(set1)
    if len(diff) > 0:
        print(f'in {df2name} but not {df1name}:')
        print(f'{diff}')
        df2_to_drop = list(diff)
    else:
        print(f'no columns in {df2name} that are not in {df1name}')
        
    return df1_to_drop, df2_to_drop

##############################################################################################################

def check_cats_match(df1, df2, df1name, df2name):
    
    """Check that values in categorical columns are in both prior to one-hot encoding"""
    
    df1_to_drop = {}
    df2_to_drop = {}
    for col in df1.columns:
        set1 = set(df1[col].unique())
        set2 = set(df2[col].unique())
        diff = set1.difference(set2)
        if len(diff) > 0:
            df1_to_drop[col] = list(diff)
        diff = set2.difference(set1)
        if len(diff) > 0:
            df2_to_drop[col] = list(diff)
            
    print(f'In {df1name} but not in {df2name}:')
    print(df1_to_drop)
    print(f'In {df2name} but not in {df1name}:')
    print(df2_to_drop)
    
    return df1_to_drop, df2_to_drop

##############################################################################################################
##################################### FACTOR ANALYSIS ########################################################
##############################################################################################################

def plot_famd(famd_results, famd_object, df, analyse_loadings=True):
    
    """plot results and loadings of FAMD"""

    print('FAMD for mixed data')
    print('\r')
    
    results = famd_results.row_coordinates(df)
    results.columns = ['PC1', 'PC2']
    
    plt.scatter(results['PC1'], results['PC2'], alpha=0.5);
    plt.xlabel('Principle Component 1 Scores');
    plt.ylabel('Principle Component 2 Scores');
    plt.show()
    
    print(f'First two components explain: {famd_object.explained_inertia_}')
    
    if analyse_loadings == True:

        eigenvectors = pd.DataFrame(famd_object.column_correlations(df))
        eigenvectors.columns=['Influence_PC1', 'Influence_PC2']

        ## Top features for PC1:

        resultsPC1 = pd.DataFrame(eigenvectors.sort_values(by='Influence_PC1', ascending=True).head(5).index)
        resultsPC1.columns = ['Driving highly negative PC1 values']
        resultsPC1['Driving highly positive PC1 values'] = eigenvectors.sort_values(by='Influence_PC1', ascending=False).head(5).index
        print(resultsPC1.head(5))
        print('\r')
    
        ## Top features for PC2:

        resultsPC1 = pd.DataFrame(eigenvectors.sort_values(by='Influence_PC2', ascending=True).head(5).index)
        resultsPC1.columns = ['Driving highly negative PC2 values']
        resultsPC1['Driving highly positive PC2 values'] = eigenvectors.sort_values(by='Influence_PC2', ascending=False).head(5).index
        print(resultsPC1.head(5))

##############################################################################################################

def plot_mca(mca_object, df, fit=True, analyse_loadings=True):
    
    """plot results and loadings of MCA"""
    
    print('MCA for categorical data')
    print('\r')
    
    if fit==True:
        results = mca_object.row_coordinates(df)
    else:
        results = mca_object
        
    results.columns = ['PC1', 'PC2']

    plt.scatter(results['PC1'], results['PC2'], alpha=0.5);
    plt.xlabel('Principle Component 1 Scores');
    plt.ylabel('Principle Component 2 Scores');
    plt.show()

    
    # print(f'First two components explain: {mca_object.explained_inertia_}, total inertia = {mca_object.total_inertia_}')
    # print('\r')
    
    if analyse_loadings == True:

        eigenvectors = pd.DataFrame(mca_object.column_coordinates(df))
        eigenvectors.columns=['Influence_PC1', 'Influence_PC2']

        ## Top features for PC1:

        resultsPC1 = pd.DataFrame(eigenvectors.sort_values(by='Influence_PC1', ascending=True).head(5).index)
        resultsPC1.columns = ['Driving highly negative PC1 values']
        resultsPC1['Driving highly positive PC1 values'] = eigenvectors.sort_values(by='Influence_PC1', ascending=False).head(5).index
        print(resultsPC1.head(5))
        print('\r')

        ## Top features for PC2:

        resultsPC1 = pd.DataFrame(eigenvectors.sort_values(by='Influence_PC2', ascending=True).head(5).index)
        resultsPC1.columns = ['Driving highly negative PC2 values']
        resultsPC1['Driving highly positive PC2 values'] = eigenvectors.sort_values(by='Influence_PC2', ascending=False).head(5).index
        print(resultsPC1.head(5))
        
    return results
        
##############################################################################################################

def plot_pca(pca_object, df, fit=True, analyse_loadings=True):
    
    """plot results and loadings of PCA"""
    
    print('PCA for categorical data')
    print('\r')
    
    if fit==True:
        results = pca_object.row_coordinates(df)
    else:
        results = pca_object
    results.columns = ['PC1', 'PC2']
    
    plt.scatter(results['PC1'], results['PC2'], alpha=0.5);
    plt.xlabel('Principle Component 1 Scores');
    plt.ylabel('Principle Component 2 Scores');
    plt.show()
    
    if analyse_loadings == True:
        print(f'First two components explain: {pca_object.explained_inertia_}, total inertia = {pca_object.total_inertia_}')
        print('\r')

        eigenvectors = pd.DataFrame(pca_object.column_correlations(df))
        eigenvectors.columns=['Influence_PC1', 'Influence_PC2']

        ## Top features for PC1:

        resultsPC1 = pd.DataFrame(eigenvectors.sort_values(by='Influence_PC1', ascending=True).head(5).index)
        resultsPC1.columns = ['Driving highly negative PC1 values']
        resultsPC1['Driving highly positive PC1 values'] = eigenvectors.sort_values(by='Influence_PC1', ascending=False).head(5).index
        print(resultsPC1.head(5))
        print('\r')

        ## Top features for PC2:

        resultsPC1 = pd.DataFrame(eigenvectors.sort_values(by='Influence_PC2', ascending=True).head(5).index)
        resultsPC1.columns = ['Driving highly negative PC2 values']
        resultsPC1['Driving highly positive PC2 values'] = eigenvectors.sort_values(by='Influence_PC2', ascending=False).head(5).index
        print(resultsPC1.head(5))
        
    return results
        
##############################################################################################################
####################################### CLUSTERING ###########################################################
##############################################################################################################

def DBSCAN_processing(clustering, df):
    
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    
    labels = clustering.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    
    ## Plot
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = df[class_member_mask & core_samples_mask]
        plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = df[class_member_mask & ~core_samples_mask]
        plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()