import pandas as pd

def load_data():
    """
    Load and preprocess the student data using COMPLETE-CASE analysis .
    Rows with any missing values in `required_vars` are dropped.
    Returns:
        pd.DataFrame: Preprocessed DataFrame (complete cases only).
    """
    path = "../../data/student_data_2018.csv"
    df = pd.read_csv(path)

    required_vars = [
        # outcome components
        "math", "read", "science",
        # treatment and key covariates
        "computer", "gender", "mother_educ", "father_educ",
        "desk", "room", "book", "country",'escs']

    # Drop rows with any missing values in required
    df_cc = df.dropna(subset=required_vars).copy()
    binary_maps = {
        "computer": {"yes": 1, "no": 0},
        "gender":   {"male": 1, "female": 0},
        "desk":     {"yes": 1, "no": 0},

        "room":     {"yes": 1, "no": 0},
    }
    for col, mapping in binary_maps.items():
        if col in df_cc.columns:
            df_cc[col] = df_cc[col].map(mapping)

    # Ordinal education
    isced_order = {
        "less than ISCED1": 0,
        "ISCED 1": 1,
        "ISCED 2": 2,
        "ISCED 3B, C": 3,
        "ISCED 3A": 4,
    }
    for col in ["mother_educ", "father_educ"]:
        if col in df_cc.columns:
            df_cc[col] = df_cc[col].map(isced_order).astype("Int64")

    # Ordinal books
    book_order = {
        "0-10": 0, "11-25": 1, "26-100": 2,
        "101-200": 3, "201-500": 4, "more than 500": 5
    }
    if "book" in df_cc.columns:
        df_cc["book"] = df_cc["book"].map(book_order).astype("Int64")

    # One-hot encode country
    if "country" in df_cc.columns:
        df_cc = pd.get_dummies(df_cc, columns=["country"], drop_first=True, dtype=int)


    if all(c in df_cc.columns for c in ["math", "read", "science"]):
        df_cc["achievement"] = df_cc[["math", "read", "science"]].mean(axis=1)


    if "computer" in df_cc.columns:
        df_cc["computer"] = df_cc["computer"].astype(int)

    return df_cc
