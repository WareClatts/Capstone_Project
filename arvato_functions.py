import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

continuous_vars = ['GEBURTSJAHR', 'KBA13_ANZAHL_PKW', 'ANZ_HAUSHALTE_AKTIV', 'ANZ_HH_TITEL',
              'ANZ_PERSONEN', 'ANZ_TITEL', 'ANZ_STATISTISCHE_HAUSHALTE', 'EINGEZOGENAM_HH_JAHR']#, 'MIN_GEBAEUDEJAHR'

##############################################################################################################
#################################### DATA PREPARATION ########################################################
##############################################################################################################

def clean_arvato_data(df):
    """
    Function specifically for cleaning the population and customer dataframes
        
    1. Column(s) that can be changed following investigation:
    
        ALTER_KINDX - binned into new categorical column denoting whether:
            0 - no children
            1 - child(ren) under 5
            2 - child(ren) under 5 - 10
            3 - child(ren) under 10 - 15
            4 - child(ren) under 15 - 20
            5 - child(ren) over 20
        
        D19_BANKEN_ONLINE_QUOTE_12:
            fill null values with 0 if D19_BANKEN_ONLINE_DATUM = 6, 7, 8, 9, 10
            
        D19_GESAMT_ONLINE_QUOTE_12:
            fill null values with 0 if D19_GESAMT_ONLINE_DATUM = 6, 7, 8, 9, 10
            
        D19_VERSAND_ONLINE_QUOTE_12:
            fill null values with 0 if D19_VERSAND_ONLINE_DATUM = 6, 7, 8, 9, 10
       
    2. Columns that may be useful have nulls pushed to existing unknown category:
        
        D19_KONSUMTYP
        'KBA05_' columns
        REGIOTYP
        KKK
        MOBI_REGIO
        'PLZ8_' columns
        'KBA13_' columns
        RELAT_AB
        ORTSGR_KLS9
        INNENSTADT
        EWDICHTE
        BALLRAUM
        ALTER_HH
        CAMEO_DEUG_2015
        EWDICHTE
        KK_KUNDENTYP
        ALTERSKATEGORIE_FEIN, assuming that like ALTERSKATEGORIE_GROSS, -1 or 0 = unknown
        
    3. Any columns that at this point are more than 50% null are dropped
    
    4. Any rows that are more than 50% null are dropped
    
    5. Otherwise they are filled with the mode (if no unknown category available):
        D19_KONSUMTYP
        MOBI_REGIO
        PLZ8_BAUMAX
        
    6. 'Unknown' is sorted out so that only one category for each column represents unknown (-1)
    
    7. 'Mixed type' columns are fixed:
        CAMEO_DEUG_2015
        CAMEO_INTL_2015
        
    8. Categorical columns converted to categorical type:
        
    
    """
    ######## 1 ######## REWORK SOME COLUMNS TO ADDRESS NULLS
    
    try: # if cleaning has been done previously
        cols_to_use = ['ALTER_KIND1', 'ALTER_KIND2', 'ALTER_KIND3', 'ALTER_KIND4']
        df['youngest_child'] = df.loc[:, cols_to_use].min(axis=1)
        df['youngest_child'].fillna(value=0, inplace=True)
        df['youngest_child_bin'] = np.NaN
        df.loc[(df['youngest_child'] == 0), 'youngest_child_bin'] = 0
        df.loc[((df['youngest_child'] > 0) & (df['youngest_child'] < 5)), 'youngest_child_bin'] = 1
        df.loc[((df['youngest_child'] >= 5) & (df['youngest_child'] < 10)), 'youngest_child_bin'] = 2
        df.loc[((df['youngest_child'] >= 10) & (df['youngest_child'] < 15)), 'youngest_child_bin'] = 3
        df.loc[((df['youngest_child'] >= 15) & (df['youngest_child'] < 20)), 'youngest_child_bin'] = 4
        df.loc[(df['youngest_child'] >= 20), 'youngest_child_bin'] = 5
        df.drop(labels=cols_to_use, axis=1, inplace=True)
        print(f'{cols_to_use} updated')
    except:
        pass
    
    try:
        for col, source in zip(['D19_BANKEN_ONLINE_QUOTE_12', 'D19_GESAMT_ONLINE_QUOTE_12',
                                'D19_VERSAND_ONLINE_QUOTE_12', 'D19_VERSI_ONLINE_QUOTE_12',
                               'D19_TELKO_ONLINE_QUOTE_12'],
                               ['D19_BANKEN_ONLINE_DATUM', 'D19_GESAMT_ONLINE_DATUM',
                                'D19_VERSAND_ONLINE_DATUM', 'D19_VERSI_ONLINE_DATUM',
                               'D19_TELKO_ONLINE_DATUM']):
            df.loc[(df[source].isin([6, 7, 8, 9, 10])), col] = 0
        print('D19 cols updated')
    except:
        pass
    
    ######## 2 ######## FILL NAN VALUES WITH UNKNOWN CATEGORY MAPPING
    
    df['D19_KONSUMTYP'].fillna(value=0, inplace=True)
    
    for col in df.columns:
        if 'KBA05_' in col:
            df[col].fillna(value=-1, inplace=True)
        
    for col in ['REGIOTYP', 'KKK']:
        df[col].fillna(value=-1, inplace=True) 
    for col in df.columns:
        if ('PLZ8_' in col) & (col != 'PLZ8_BAUMAX'):
            df[col].fillna(value=-1, inplace=True)
        elif 'KBA13_' in col:
            df[col].fillna(value=-1, inplace=True)
            
    for col in ['RELAT_AB', 'ORTSGR_KLS9', 'INNENSTADT', 'EWDICHTE', 'BALLRAUM', 'CAMEO_DEUG_2015',
                'EWDICHTE', 'KK_KUNDENTYP', 'ALTERSKATEGORIE_FEIN']:
        df[col].fillna(value=-1, inplace=True) 
        
    df['ALTER_HH'].fillna(value=0, inplace=True)
    
    print('NaNs pushed to unknown where possible')
    
    ######## 3 ######## REMOVE COLUMNS WITH >= 50% NULLS
    
    for col in df.columns:
        if df[col].isna().sum() >0:
            if (df[col].isna().sum())/(df[col].count()) >= 0.5:
                print(f'column {col} dropped due to >50% nulls')
                df.drop(col, axis=1, inplace=True)
    
    ######## 4 ######## REMOVE ROWS WITH >= 50% NULLS
    
    previous_rows = df.shape[1]
    threshold = df.shape[1]/2
    df.dropna(axis=0, thresh=threshold)
    print(f'{previous_rows-df.shape[1]} majority-null rows dropped')
    
    ######## 5 ######## IMPUTE REMAINING NULLS BY IMPUTATION USING COLUMN MODE (better for categorical than mean!)
    
    i=0
    for col in df.columns:
        if df[col].isna().sum() >0:
            df.fillna(value=df[col].mode()[0], inplace=True)
            i+=1
    print(f'{i} columns had nulls imputed with mode')
    
    ######## 6 ######## ENSURE UNKNOWN IS ONE CATEGORY ONLY (-1)

    # Columns with -1, 0 for unknown: #
    columns0 = ['ALTERSKATEGORIE_GROB', 'ANREDE_KZ', 'GEBAEUDETYP', 'HH_EINKOMMEN_SCORE', 
                'KBA05_BAUMAX', 'KBA05_GBZ', 'KKK', 'NATIONALITAET_KZ', 'PRAEGENDE_JUGENDJAHRE', 'REGIOTYP', 'TITEL_KZ', 
                'WOHNDAUER_2008', 'GEOSCORE_KLS7', 'HAUSHALTSSTRUKTUR', 'WACHSTUMSGEBIET_NB', 'W_KEIT_KIND_HH']
    for col in columns0:
        # in case column has been dropped previously:
        try:
            df.loc[(df[col] == 0), col] = -1
        except Exception as inst:
            print(f'cannot correct {col}:')
            print(f'    {type(inst)}')       # the exception instance
            print(f'    {inst.args}')        # arguments stored in .args
            print(f'    {inst}') 
            

    # Columns with -1, 9 for unknown:
    columns9 = ['KBA05_ALTER1','KBA05_ALTER2','KBA05_ALTER3','KBA05_ALTER4','KBA05_ANHANG','KBA05_AUTOQUOT','KBA05_CCM1',
                'KBA05_CCM2','KBA05_CCM3','KBA05_CCM4','KBA05_DIESEL','KBA05_FRAU','KBA05_HERST1','KBA05_HERST2','KBA05_HERST3',
                'KBA05_HERST4','KBA05_HERST5','KBA05_HERSTTEMP','KBA05_KRSAQUOT','KBA05_KRSHERST1','KBA05_KRSHERST2',
                'KBA05_KRSHERST3','KBA05_KRSKLEIN','KBA05_KRSOBER','KBA05_KRSVAN','KBA05_KRSZUL','KBA05_KW1','KBA05_KW2',
                'KBA05_KW3','KBA05_MAXAH','KBA05_MAXBJ','KBA05_MAXHERST','KBA05_MAXSEG','KBA05_MAXVORB','KBA05_MOD1',
                'KBA05_MOD2','KBA05_MOD3','KBA05_MOD4','KBA05_MOD8','KBA05_MODTEMP','KBA05_MOTOR','KBA05_MOTRAD','KBA05_SEG1',
                'KBA05_SEG10','KBA05_SEG2','KBA05_SEG3','KBA05_SEG4','KBA05_SEG5','KBA05_SEG6','KBA05_SEG7','KBA05_SEG8',
                'KBA05_SEG9','KBA05_VORB0','KBA05_VORB1','KBA05_VORB2','KBA05_ZUL1','KBA05_ZUL2','KBA05_ZUL3','KBA05_ZUL4',
                'RELAT_AB','SEMIO_DOM','SEMIO_ERL','SEMIO_FAM','SEMIO_KAEM','SEMIO_KRIT','SEMIO_KULT','SEMIO_LUST','SEMIO_MAT',
                'SEMIO_PFLICHT','SEMIO_RAT','SEMIO_REL','SEMIO_SOZ','SEMIO_TRADV','SEMIO_VERT','ZABEOTYP']
    for col in columns9:
        # in case column has been dropped previously:
        try:
            df.loc[(df[col] == 9), col] = -1
        except Exception as inst:
            print(f'cannot correct {col}:')
            print(f'    {type(inst)}')       # the exception instance
            print(f'    {inst.args}')        # arguments stored in .args
            print(f'    {inst}') 
    
    print('Unknowns forced to single category')
    
    ######## 7 ######## FIX MIXED DATA TYPES (SEE ERROR ON IMPORT)
    
    # in case of multiple partial passes through cleaning during development:
    try:
        df.loc[(df['CAMEO_DEUG_2015'] == 'X'), 'CAMEO_DEUG_2015'] = -1
    except:
        pass
    try:
        df.loc[(df['CAMEO_INTL_2015'] == 'XX'), 'CAMEO_INTL_2015'] = -1
    except:
        pass
    for col in ['CAMEO_DEUG_2015','CAMEO_INTL_2015']:
        df[col].fillna(value=-1, inplace=True) # may not be necessary not sure
        df[col] = [int(x) for x in df[col]] 
        
    print('Mixed data types fixed')
    
    ######## 8 ######## CONVERT TO CATEGORICAL
    
    ## Easier to convert categoricals by first pulling out columns that are NOT categorical
    noncat = ['GEBURTSJAHR', 'KBA13_ANZAHL_PKW', 'MIN_GEBAEUDEJAHR']

    for col in df.columns:
        if col in noncat:
            pass
        else:
            df[col].astype('category')
    
    print('Columns converted to categorical')
    
    ## Print summary?
        
    for col in df.columns:
        if df[col].isna().sum() >0:
            print(col)
    print('finished')
    print(f'size of dataset: {df.shape}')

    return df

##############################################################################################################

def clean_arvato_data_2(df):
    """
    Function specifically for cleaning the population and customer dataframes
        
    1. Column(s) that can be changed following investigation:
    
        ALTER_KINDX - binned into new categorical column denoting whether:
            0 - no children
            1 - child(ren) under 5
            2 - child(ren) under 5 - 10
            3 - child(ren) under 10 - 15
            4 - child(ren) under 15 - 20
            5 - child(ren) over 20
        
        D19_BANKEN_ONLINE_QUOTE_12:
            fill null values with 0 if D19_BANKEN_ONLINE_DATUM = 6, 7, 8, 9, 10
            
        D19_GESAMT_ONLINE_QUOTE_12:
            fill null values with 0 if D19_GESAMT_ONLINE_DATUM = 6, 7, 8, 9, 10
            
        D19_VERSAND_ONLINE_QUOTE_12:
            fill null values with 0 if D19_VERSAND_ONLINE_DATUM = 6, 7, 8, 9, 10
       
    2. 'Unknown' is sorted out so that all unknown are NaN
        
    3. Any columns that are more than 50% null are dropped
    
    4. Any rows that are more than 50% null are dropped
    
    5. Otherwise they are filled with the mode (if no unknown category available):
        D19_KONSUMTYP
        MOBI_REGIO
        PLZ8_BAUMAX
    
    7. 'Mixed type' columns are fixed:
        CAMEO_DEUG_2015
        CAMEO_INTL_2015
        
    8. Categorical columns converted to categorical type:
        
    
    """
    ######## 1 ######## REWORK SOME COLUMNS TO ADDRESS NULLS
    
    try: # if cleaning has been done previously
        cols_to_use = ['ALTER_KIND1', 'ALTER_KIND2', 'ALTER_KIND3', 'ALTER_KIND4']
        df['youngest_child'] = df.loc[:, cols_to_use].min(axis=1)
        df['youngest_child'].fillna(value=0, inplace=True)
        df['youngest_child_bin'] = np.NaN
        df.loc[(df['youngest_child'] == 0), 'youngest_child_bin'] = 0
        df.loc[((df['youngest_child'] > 0) & (df['youngest_child'] < 5)), 'youngest_child_bin'] = 1
        df.loc[((df['youngest_child'] >= 5) & (df['youngest_child'] < 10)), 'youngest_child_bin'] = 2
        df.loc[((df['youngest_child'] >= 10) & (df['youngest_child'] < 15)), 'youngest_child_bin'] = 3
        df.loc[((df['youngest_child'] >= 15) & (df['youngest_child'] < 20)), 'youngest_child_bin'] = 4
        df.loc[(df['youngest_child'] >= 20), 'youngest_child_bin'] = 5
        df.drop(labels=cols_to_use, axis=1, inplace=True)
        print(f'{cols_to_use} updated')
    except:
        pass
    
    try:
        for col, source in zip(['D19_BANKEN_ONLINE_QUOTE_12', 'D19_GESAMT_ONLINE_QUOTE_12',
                                'D19_VERSAND_ONLINE_QUOTE_12', 'D19_VERSI_ONLINE_QUOTE_12',
                               'D19_TELKO_ONLINE_QUOTE_12'],
                               ['D19_BANKEN_ONLINE_DATUM', 'D19_GESAMT_ONLINE_DATUM',
                                'D19_VERSAND_ONLINE_DATUM', 'D19_VERSI_ONLINE_DATUM',
                               'D19_TELKO_ONLINE_DATUM']):
            df.loc[(df[source].isin([6, 7, 8, 9, 10])), col] = 0
        print('D19 cols updated')
    except:
        pass
    
    ######## 2 ######## MOVE UNKNOWN TO NAN
    
    # Columns with -1, 0 for unknown: #
    columns0 = ['ALTERSKATEGORIE_GROB', 'ANREDE_KZ', 'GEBAEUDETYP', 'HH_EINKOMMEN_SCORE', 
                'KBA05_BAUMAX', 'KBA05_GBZ', 'KKK', 'NATIONALITAET_KZ', 'PRAEGENDE_JUGENDJAHRE', 'REGIOTYP', 'TITEL_KZ', 
                'WOHNDAUER_2008', 'GEOSCORE_KLS7', 'HAUSHALTSSTRUKTUR', 'WACHSTUMSGEBIET_NB', 'W_KEIT_KIND_HH']
    for col in columns0:
        # in case column has been dropped previously:
        try:
            df.loc[(df[col] == 0), col] = np.nan
        except:
            print(f'column {col} cannot be filled with nan and is dropped')
            df.drop(col, axis=1, inplace=True)
            

    # Columns with -1, 9 for unknown:
    columns9 = ['KBA05_ALTER1','KBA05_ALTER2','KBA05_ALTER3','KBA05_ALTER4','KBA05_ANHANG','KBA05_AUTOQUOT','KBA05_CCM1',
                'KBA05_CCM2','KBA05_CCM3','KBA05_CCM4','KBA05_DIESEL','KBA05_FRAU','KBA05_HERST1','KBA05_HERST2','KBA05_HERST3',
                'KBA05_HERST4','KBA05_HERST5','KBA05_HERSTTEMP','KBA05_KRSAQUOT','KBA05_KRSHERST1','KBA05_KRSHERST2',
                'KBA05_KRSHERST3','KBA05_KRSKLEIN','KBA05_KRSOBER','KBA05_KRSVAN','KBA05_KRSZUL','KBA05_KW1','KBA05_KW2',
                'KBA05_KW3','KBA05_MAXAH','KBA05_MAXBJ','KBA05_MAXHERST','KBA05_MAXSEG','KBA05_MAXVORB','KBA05_MOD1',
                'KBA05_MOD2','KBA05_MOD3','KBA05_MOD4','KBA05_MOD8','KBA05_MODTEMP','KBA05_MOTOR','KBA05_MOTRAD','KBA05_SEG1',
                'KBA05_SEG10','KBA05_SEG2','KBA05_SEG3','KBA05_SEG4','KBA05_SEG5','KBA05_SEG6','KBA05_SEG7','KBA05_SEG8',
                'KBA05_SEG9','KBA05_VORB0','KBA05_VORB1','KBA05_VORB2','KBA05_ZUL1','KBA05_ZUL2','KBA05_ZUL3','KBA05_ZUL4',
                'RELAT_AB','SEMIO_DOM','SEMIO_ERL','SEMIO_FAM','SEMIO_KAEM','SEMIO_KRIT','SEMIO_KULT','SEMIO_LUST','SEMIO_MAT',
                'SEMIO_PFLICHT','SEMIO_RAT','SEMIO_REL','SEMIO_SOZ','SEMIO_TRADV','SEMIO_VERT','ZABEOTYP']
    for col in columns9:
        # in case column has been dropped previously:
        try:
            df.loc[(df[col] == 9), col] = np.nan
        except:
            print(f'column {col} cannot be filled with nan and is dropped')
            df.drop(col, axis=1, inplace=True)
    
    print('Unknowns forced to NaN')
    
    ######## 3 ######## REMOVE COLUMNS WITH >= 50% NULLS
    
    for col in df.columns:
        if df[col].isna().sum() >0:
            if (df[col].isna().sum())/(df[col].count()) >= 0.5:
                print(f'column {col} dropped due to >50% nulls')
                df.drop(col, axis=1, inplace=True)
    
    ######## 4 ######## REMOVE ROWS WITH >= 50% NULLS
    
    previous_rows = df.shape[1]
    threshold = df.shape[1]/2
    df.dropna(axis=0, thresh=threshold)
    print(f'{previous_rows-df.shape[1]} majority-null rows dropped')
    
    ######## 5 ######## IMPUTE REMAINING NULLS BY IMPUTATION USING COLUMN MODE (better for categorical than mean!)
    
    i=0
    for col in df.columns:
        if df[col].isna().sum() >0:
            df.fillna(value=df[col].mode()[0], inplace=True)
            i+=1
    print(f'{i} columns had nulls imputed with mode')
    
    ######## 7 ######## FIX MIXED DATA TYPES (SEE ERROR ON IMPORT)
    
    # in case of multiple partial passes through cleaning during development:
    try:
        df.loc[(df['CAMEO_DEUG_2015'] == 'X'), 'CAMEO_DEUG_2015'] = -1
    except:
        pass
    try:
        df.loc[(df['CAMEO_INTL_2015'] == 'XX'), 'CAMEO_INTL_2015'] = -1
    except:
        pass
    for col in ['CAMEO_DEUG_2015','CAMEO_INTL_2015']:
        df[col].fillna(value=-1, inplace=True) # may not be necessary not sure
        df[col] = [int(x) for x in df[col]] 
        
    print('Mixed data types fixed')
    
    ######## 8 ######## CONVERT TO CATEGORICAL
    
    ## Easier to convert categoricals by first pulling out columns that are NOT categorical
    noncat = ['GEBURTSJAHR', 'KBA13_ANZAHL_PKW', 'MIN_GEBAEUDEJAHR']

    for col in df.columns:
        if col in noncat:
            pass
        else:
            df[col].astype('category')
    
    print('Columns converted to categorical')
    
    ## Print summary?
        
    for col in df.columns:
        if df[col].isna().sum() >0:
            print(col)
    print('finished')
    print(f'size of dataset: {df.shape}')

    return df

##############################################################################################################

def tidy_data(df):   
    ## Misc columns and indexing (unless previously done):
    
    # EINGEFUEGT_AM - some sort of timestamp, not needed
    df.drop(labels=['EINGEFUEGT_AM'], axis=1, inplace=True)
    
    ## Date house appeared in database - not useful
    df.drop(labels=['MIN_GEBAEUDEJAHR'], axis=1, inplace=True)

    # UNAMED - a reset index from writing - can be dropped
    df.drop(labels=['Unnamed: 0'], axis=1, inplace=True)

    # Also set the index to LNR (the id)
    df.set_index('LNR', inplace=True)
    
    return df

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

def prepare_mca(df, to_drop, continuous_vars, reduce_columns, subsample=0.5, onehot=True):
    
    """Preparing data for MCA"""
    
    try:
        test = df.sample(frac=subsample, replace=False, random_state=1).set_index('LNR')
    except:
        test = df.sample(frac=subsample, replace=False, random_state=1)
    
    test.drop(labels=continuous_vars, axis=1, inplace=True)
    
    if reduce_columns == True:
        test.drop(labels=to_drop, axis=1, inplace=True)
        
    if onehot==True:
        test = prepare_cat_cols(test, continuous_vars, 'all_cat_cols') 
    else:
        test = prepare_cat_cols(test, continuous_vars, 'None') 
    
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

def match_ohe_dfs(df1, df2, df1name='df1', df2name='df2'):
    
    """Note that df2 is not changed, only df1 is changed to match df2 by dropping columns after ohe 
    and adding dummy columns filled with """
    
    print(f'Checking {df1name} and {df2name} match...')
    todrop1, todrop2 = check_cols_match(df1, df2, df1name, df2name)
    print('\r')

    while (len(todrop1) > 0) | (len(todrop2) > 0):

        df1.drop(labels=todrop1, axis=1, inplace=True)
        for col in todrop2:
            df1[col] = 0

        todrop1, todrop2 = check_cols_match(df1, df2, df1name, df2name)
    
    print(f'...{df1name} and {df2name} now match')
    
    print(f'Matching column order...')
    df1 = df1[df2.columns.tolist()]
    print('...done')
        
    return df1, df2

##############################################################################################################

from sklearn.model_selection import train_test_split

def get_matching_ohe_traintest(X, y, to_drop, continuous_vars, scaler, test_size=0.33, random_state=1, scale=True):
    
    print('Getting train test split...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    print('...done')
    # print('\r')
    
    print('One-hot encoding categorical X data and dropping columns to drop...')
    X_train_ohe = prep_classification(X_train, to_drop, continuous_vars, reduce_columns=True, subsample=1.0, onehot=True)
    X_test_ohe = prep_classification(X_test, to_drop, continuous_vars, reduce_columns=True, subsample=1.0, onehot=True)
    print('...done')
    # print('\r')
    
    X_train_ohe, X_test_ohe = match_ohe_dfs(X_train_ohe, X_test_ohe, 'train', 'test')
    
    # print('\r')
    
    ## Scale the data
    if scale==True:
        print('Scaling data to train...')

        Xtrain = scaler.fit_transform(X_train_ohe)
        Xtest = scaler.transform(X_test_ohe)
        print('...done')
        # print('\r')
    
        print(f'Training dataset = {Xtrain.shape}, testing dataset = {Xtest.shape}')
        return Xtrain, Xtest, y_train, y_test
    
    print(f'Training dataset = {X_train_ohe.shape}, testing dataset = {X_test_ohe.shape}')
    return X_train_ohe, X_test_ohe, y_train, y_test

##############################################################################################################
######################################### CHECKS #############################################################
##############################################################################################################

def nullcols_check(df):
    df_nullcols =  pd.DataFrame(data = df.isnull().sum(axis=0))
    df_nullcols['pc_rows_null'] = (df_nullcols.iloc[:,0]/df.shape[0])*100
    
    plt.hist(df_nullcols['pc_rows_null']);
    plt.title('How many mostly null columns in df')
    plt.xlabel('Percentage of null rows per columns')
    plt.ylabel('Frequency')
    plt.show()
    
    return df_nullcols

##############################################################################################################

def nullrows_check(df):
    df_nullrows =  pd.DataFrame(data = df.isnull().sum(axis=1))
    df_nullrows['pc_columns_null'] = (df_nullrows.iloc[:,0]/df.shape[1])*100
    
    plt.hist(df_nullrows['pc_columns_null']);
    plt.title('How many mostly null rows in df')
    plt.xlabel('Percentage of null entries per row')
    plt.ylabel('Frequency')
    plt.show()
    
    return df_nullrows

##############################################################################################################

def unknown_check(df):
    df_ukcols =  pd.DataFrame(data = df.eq(-1).sum(axis=0))
    df_ukcols['pc_rows_unknown'] = (df_ukcols.iloc[:,0]/df.shape[0])*100
    
    plt.hist(df_ukcols['pc_rows_unknown']);
    plt.title('How many mostly null columns in df')
    plt.xlabel('Percentage of unknown rows per columns')
    plt.ylabel('Frequency')
    plt.show()
    
    return df_ukcols

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

##################################only works in main script###################################################

# def get_df_name(df):
#     name =[x for x in globals() if globals()[x] is df][0]
#     return name

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

def plot_mca(mca_object, df, fit=True, analyse_loadings=True, show_inertia=False):
    
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

    if show_inertia==True:
        print(f'First two components explain: {mca_object.explained_inertia_}, total inertia = {mca_object.total_inertia_}')
        print('\r')
    
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
                 markeredgecolor='k', markersize=14, label=f'{k}')

        xy = df[class_member_mask & ~core_samples_mask]
        plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.legend(loc='best')
    plt.show()
    
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def clusters_3dplot(df, cluster_col, x_col, y_col, z_col, cluster_labels):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    scale = 8
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(cluster_labels))]

    # Make data.
    for i in cluster_labels:

        temp = df[df[cluster_col]==i]
        Z = temp[z_col]
        X = temp[x_col]
        Y = temp[y_col]

        # Plot the surface.
        if i==-1:
            color='k'
        else:
            color=colors[i]
        scat = ax.scatter(X, Y, Z, c=color, linewidth=0, antialiased=False)

    ax.set_xlabel('MCA component 1')
    ax.set_ylabel('MCA component 2')
    ax.set_zlabel('PCA component 1')

    # rotate the axes and update
    for angle in range(0, 360):
        ax.view_init(30, 40)

    plt.show()
    
##############################################################################################################
######################################## MODELING ############################################################
##############################################################################################################


def prep_classification(df, to_drop, continuous_vars, reduce_columns=True, subsample=1.0, onehot=True):
    
    """Prepare data for classification algorithms"""
        
    try:
        test = df.sample(frac=subsample, replace=False, random_state=1).set_index('LNR')
    except:
        test = df.sample(frac=subsample, replace=False, random_state=1)
    
    ## Drop any features not wanted
    if reduce_columns == True:
        test.drop(labels=to_drop, axis=1, inplace=True)
        
    ## One-hot encode categorical features
    if onehot==True:
        test = prepare_cat_cols(test, continuous_vars, 'all_cat_cols') 
    else:
        test = prepare_cat_cols(test, continuous_vars, 'string_cols') #None 
    
    print(f'shape of prepared data = {test.shape[0]} rows, {test.shape[1]} columns')
    
    return test

##############################################################################################################

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from timeit import default_timer as timer

def test_models(model_list, model_names, Xtrain, y_train, Xtest, y_test, printout=True):
    accs = []
    f1s = []
    precisions = []
    recalls = []
    aucs = []
    runtimes = []

    for model, name in zip(model_list, model_names):
        
        start = timer()
        
        print(f'Fitting model {model}')
        model.fit(Xtrain, y_train)
        pred_test = model.predict(Xtest)

        # Print f1 and accuracy scores    
        acc = accuracy_score(y_test, pred_test)
        accs.append(round(acc, 3))
        f1 = f1_score(y_test, pred_test)
        f1s.append(round(f1, 3))

        # Print the confusion matrix
        results = pd.DataFrame(confusion_matrix(y_test, pred_test))
        tp = results.iloc[1,1]
        fp = results.iloc[0,1]
        #tn = results.iloc[0,0]
        fn = results.iloc[1,0]
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        precisions.append(round(precision, 3))
        recalls.append(round(recall, 3))
        if printout==True:
            print(results)

        # Print the ROC
        try:
            y_score = model.predict_proba(Xtest)[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            aucs.append(round(roc_auc, 3))
            if printout==True:
                plt.plot(fpr, tpr)
                plt.title(f'ROC curve for {name} (area ={round(roc_auc, 4)})')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.show()
        except:
            aucs.append(np.nan)
        
        end = timer()
        elapsed = end - start
        print(f'Run time in seconds: {elapsed}')
        runtimes.append(round(elapsed))
        if printout==True:
            print('\r')

    resultsdf = pd.DataFrame(index=model_names)
    resultsdf['Accuracy'] = accs
    resultsdf['F1_scores'] = f1s
    resultsdf['Precision'] = precisions
    resultsdf['Recall'] = recalls
    resultsdf['AUC'] = aucs
    resultsdf['Runtime_s'] = runtimes

    return resultsdf