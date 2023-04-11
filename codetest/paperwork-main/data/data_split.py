from sklearn.model_selection import StratifiedKFold

def kfsplit(data_df,CONFIG,skip_info=False):

    kf = StratifiedKFold(n_splits=CONFIG["NUM_FOLDS"], shuffle=True, random_state=CONFIG["SEED"])

    for f, (t_, v_) in enumerate(kf.split(data_df, data_df.label)):
        data_df.loc[v_, 'fold'] = f
    if skip_info==False:

        CONFIG["Logger"].info("-"*10+"data_df fold value_counts"+"-"*10)
        CONFIG["Logger"].info(data_df.fold.value_counts())

    return data_df

if __name__=='__main__':
    pass