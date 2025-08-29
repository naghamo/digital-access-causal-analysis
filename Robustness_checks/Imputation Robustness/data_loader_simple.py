import pandas as pd
import miceforest as mf
def load_data():
    """
    Load and preprocess the student data with missing value imputation.
    Uses mean/mode missingness
    Returns:
        pd.DataFrame: Preprocessed DataFrame with imputed values.
    """
    df = pd.read_csv("../../data/student_data_2018.csv")


    # Calculate missingness per column
    missing_percent = df.isnull().mean()
    missing_cols = missing_percent.index.tolist()


    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    # Mean imputation for numeric columns with low missingness
    mean_impute_cols = [col for col in missing_cols if col in numeric_cols]
    df_mean_imputed = df[mean_impute_cols].copy()
    for col in mean_impute_cols:
        df_mean_imputed[col] = df_mean_imputed[col].fillna(df_mean_imputed[col].mean())


    # Mode imputation for non-numeric columns with low missingness
    mode_impute_cols = [col for col in missing_cols if col in non_numeric_cols]
    df_mode_imputed = df[mode_impute_cols].copy()
    for col in mode_impute_cols:
        if df_mode_imputed[col].isnull().any():
            df_mode_imputed[col] = df_mode_imputed[col].fillna(df_mode_imputed[col].mode().iloc[0])

    # Concatenate all imputed columns
    all_imputed = pd.concat(
        [df_mean_imputed, df_mode_imputed], axis=1
    )

    # Reindex to original column order and fill any remaining gaps from the original df
    df_final = df.copy()
    for col in all_imputed.columns:
        df_final[col] = all_imputed[col]
    df_final['achievement'] = df_final[['math', 'read', 'science']].mean(axis=1)
    df_final['computer'] = df_final['computer'].map({'yes': 1, 'no': 0}).fillna(df_final['computer'])
    df_final['gender'] = df_final['gender'].map({'male': 1, 'female': 0}).fillna(df_final['gender'])
    df_final['read'] = df_final['read'].fillna(df_final['read'].mean())
    df_final['computer'] = df_final['computer'].astype(int)
    isced_order = {
        'less than ISCED1': 0,
        'ISCED 1': 1,
        'ISCED 2': 2,
        'ISCED 3B, C': 3,
        'ISCED 3A': 4
    }

    for col in ['mother_educ', 'father_educ']:
        df_final[col] = df_final[col].map(isced_order).astype('Int64')
    df_final = pd.get_dummies(df_final, columns=['country'], drop_first=True, dtype=int)
    #['0-10' '11-25' '26-100' '101-200' 'more than 500' '201-500']
    book_order={'0-10': 0, '11-25': 1, '26-100': 2, '101-200': 3, '201-500': 4, 'more than 500': 5}
    df_final['book'] = df_final['book'].map(book_order).fillna(df_final['book'])
    df_final['desk']= df_final['desk'].map({'yes': 1, 'no': 0}).fillna(df_final['desk'])
    df_final['internet'] = df_final['internet'].map({'yes': 1, 'no': 0}).fillna(df_final['internet'])
    df_final['room'] = df_final['room'].map({'yes': 1, 'no': 0}).fillna(df_final['room'])

    return df_final
