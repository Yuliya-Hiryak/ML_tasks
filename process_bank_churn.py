import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Any

def drop_na_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NA values in the specified columns.

    Args:
        df (pd.DataFrame): The raw dataframe.
        columns (list): List of columns to check for NA values.

    Returns:
        pd.DataFrame: DataFrame with NA values dropped.
    """
    # find n/a values in df
    columns = df.columns[df.isna().any()].tolist()
    if not columns:
        return df

    return df.dropna(subset=columns)

def split_data(df: pd.DataFrame, target_col: str) -> Dict[str, pd.DataFrame]:
    """
    Split the dataframe into training, validation, and test sets based on the year.

    Args:
        df (pd.DataFrame): The raw dataframe.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing the train, validation, and test dataframes.
    """
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    return {'train': train_df, 'val': val_df}

def create_inputs_targets(df_dict: Dict[str, pd.DataFrame], input_cols: list, target_col: str) -> Dict[str, Any]:
    """
    Create inputs and targets for training, and validation sets.

    Args:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing the train, validation, and test dataframes.
        input_cols (list): List of input columns.
        target_col (str): Target column.

    Returns:
        Dict[str, Any]: Dictionary containing inputs and targets for train, and val sets.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data

def impute_missing_values(data: Dict[str, Any], numeric_cols: list) -> None:
    """
    Impute missing numerical values using the mean strategy.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train, val, and test sets.
        numeric_cols (list): List of numerical columns.
    """
    imputer = SimpleImputer(strategy='mean').fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = imputer.transform(data[f'{split}_inputs'][numeric_cols])

def scale_numeric_features(data: Dict[str, Any], numeric_cols: list) -> None:
    """
    Scale numeric features using MinMaxScaler.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train, and val sets.
        numeric_cols (list): List of numerical columns.
    """
    scaler = MinMaxScaler()
    scaler.fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = scaler.transform(data[f'{split}_inputs'][numeric_cols])
    return scaler

def encode_categorical_features(data: Dict[str, Any], categorical_cols: list) -> None:
    """
    One-hot encode categorical features.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train, and val sets.
        categorical_cols (list): List of categorical columns.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(data['train_inputs'][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    for split in ['train', 'val']:
        encoded = encoder.transform(data[f'{split}_inputs'][categorical_cols])
        data[f'{split}_inputs'] = pd.concat([data[f'{split}_inputs'], pd.DataFrame(encoded, columns=encoded_cols, index=data[f'{split}_inputs'].index)], axis=1)
        data[f'{split}_inputs'].drop(columns=categorical_cols, inplace=True)
    data['encoded_cols'] = encoded_cols

    return encoder

def preprocess_data(raw_df: pd.DataFrame, target_col: str, scaler_numeric: bool = True) -> Dict[str, Any]:
    """
    Preprocess the raw dataframe.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.
        target_col (str): The target column to predict.
        scaler_numeric (bool): Whether to scale numeric features.

    Returns:
        Dict[str, Any]: Dictionary containing processed inputs and targets for train, and val sets.
    """
    raw_df = drop_na_values(raw_df)
    split_dfs = split_data(raw_df, target_col)
    input_cols = [col for col in raw_df.columns if col not in [target_col, 'id', 'Surname', 'CustomerId']]
    data = create_inputs_targets(split_dfs, input_cols, target_col)

    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes('object').columns.tolist()

    impute_missing_values(data, numeric_cols)

    scaler = None
    if scaler_numeric:
        scaler = scale_numeric_features(data, numeric_cols)
    encoder = encode_categorical_features(data, categorical_cols)

    # Extract X_train, X_val
    X_train = data['train_inputs'][numeric_cols + data['encoded_cols']]
    X_val = data['val_inputs'][numeric_cols + data['encoded_cols']]

    return {
        'X_train': X_train,
        'train_targets': data['train_targets'],
        'X_val': X_val,
        'val_targets': data['val_targets'],
        "input_cols": numeric_cols + data['encoded_cols'],
        "scaler": scaler,
        "encoder": encoder
    }

def preprocess_new_data(
    new_df: pd.DataFrame,
    scaler: MinMaxScaler,
    encoder: OneHotEncoder,
    input_cols: list
) -> pd.DataFrame:
    """
    Preprocess new data using the fitted scaler and encoder.

    Args:
        new_df (pd.DataFrame): New raw dataframe.
        scaler (MinMaxScaler): Trained scaler.
        encoder (OneHotEncoder): Trained encoder.
        input_cols (list): List of final input column names from training set.

    Returns:
        pd.DataFrame: Preprocessed new dataframe ready for prediction.
    """
    df = new_df.copy()

    for col in ['Surname', 'id', 'CustomerId']:
        if col in df.columns:
            df = df.drop(columns=col)

    if scaler:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    categorical_cols = df.select_dtypes('object').columns.tolist()
    encoded = encoder.transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

    df = pd.concat([df.drop(columns=categorical_cols), df_encoded], axis=1)

    for col in input_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[input_cols]

    return df
