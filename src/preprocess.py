delta = 1.E-6
def preprocess(df, tick_size):
    for column in df.columns:
        if 'price' in column:
            df[column] = ((df[column].values+delta)/tick_size).astype(int)
        else:
            df[column] = (df[column].values*(1.+delta)).astype(int)
    return df