def filter_n_staging(x):
    if x == '1A' or x == '1a':
        return 1
    elif x == '1B' or x == '1b':
        return 2
    elif x == '2A' or x == '2a':
        return 3
    elif x == '2B' or x == '2b':
        return 4
    else:
        return 0

def filter_t_staging(x):
    if x == '4a' or x == '4A':
        return 4
    elif x == '4b' or x == '4B':
        return 5
    else:
        return int(x)

def filter_yes_no(x):
    if str.lower(x) == 'yes':
        return 1
    elif str.lower(x) == 'no':
        return 0
    else:
        return x

def filter_istologic_type(x):
    if str.lower(x) == 'adenocarcinoma':
        return 1
    elif str.lower(x) == 'adenocarcinoma ulceroso':
        return 2
    else:
        return 0

def filter_cms(x):
    if x == "U":
        return 0
    else:
        return 1
    #if x == "CMS1":
    #    return 1
    #if x == "CMS2":
    #    return 2
    #if x == "CMS3":
    #    return 3
    #if x == "CMS4":
    #    return 4
    #return 0

# cut at 1 
def filter_greater_zero(x):
    if x > 0:
        return 1
    else:
        return 0

def filter_genomics(x):
    if x == 2:
        return 1
    else:
        return 0

def remove_features(df, features):
    temp = features.copy()
    for f in temp:
        if f in df:
            df.drop(inplace=True, columns=[f])
            features.remove(f)
    return df, features

def process_genomical_features(df, features):
    # filter CMS
    if 'CMS' in df:
        df['CMS'] = df['CMS'].apply(filter_cms)
    if 'CMS' in features:
        features.remove('CMS')
    # fill with 0 missing values
    for f in features:
        df[f] = df[f].fillna(0)
    # only 2 (malign) becomes 1, rest is 0
    for f in features:
        df[f] = df[f].apply(filter_genomics)
    return df
    
def process_colon_db(df):
    # fill na
    #df = df.fillna(df.median(), inplace=True)
    # apply filters
    if 'N-STAGING' in df:
        df['N-STAGING'] = df['N-STAGING'].apply(filter_n_staging)
    if 'T-STAGING' in df:
        df['T-STAGING'] = df['T-STAGING'].apply(filter_t_staging)
    if 'LVI' in df:
        df['LVI'] = df['LVI'].fillna('no')
        df['LVI'] = df['LVI'].apply(filter_yes_no)
    if 'PNI' in df:
        df['PNI'] = df['PNI'].fillna('no')
        df['PNI'] = df['PNI'].apply(filter_yes_no)
    if 'BUDDING' in df:
        df['BUDDING'] = df['BUDDING'].fillna('no')
        df['BUDDING'] = df['BUDDING'].apply(filter_yes_no)
    if 'ISTOLOGIA' in df:
        df['ISTOLOGIA'] = df['ISTOLOGIA'].fillna('adenocarcinoma')
        #df['ISTOLOGIA'] = df['ISTOLOGIA'].fillna(df['ISTOLOGIA'].median())
        df['ISTOLOGIA'] = df['ISTOLOGIA'].apply(filter_istologic_type)
    if 'MSI' in df:    
        df.drop(inplace=True, columns=['MSI'])
    if 'NODES' in df:    
        df.drop(inplace=True, columns=['NODES'])
    return df
