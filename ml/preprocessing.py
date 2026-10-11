import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
	EMPLOYEE_ID_COLUMNS,
	EMPLOYEE_NAME_COLUMNS,
	ATTRITION_COLUMNS,
)


def find_employee_column(df):
	"""Find the employee ID or employee name column."""

	for column in EMPLOYEE_ID_COLUMNS:
		if column in df.columns:
			return column

	for column in EMPLOYEE_NAME_COLUMNS:
		if column in df.columns:
			return column

	return None


def find_attrition_column(df):
	"""Find the attrition/turnover target column."""

	for column in ATTRITION_COLUMNS:
		if column in df.columns:
			return column

	return None


def validate_dataset(df):
	"""Validate the uploaded employee dataset."""

	errors = []

	if df.empty:
		errors.append("The uploaded dataset is empty.")

	if len(df) < 20:
		errors.append(
			"The dataset contains fewer than 20 records. "
			"More data may be required for meaningful model training."
		)

	employee_column = find_employee_column(df)

	if employee_column is None:
		errors.append("No Employee ID or Employee Name column was detected.")

	attrition_column = find_attrition_column(df)

	if attrition_column is None:
		errors.append("No Attrition/Turnover target column was detected.")

	duplicate_count = df.duplicated().sum()

	return {
		"valid": len(errors) == 0,
		"errors": errors,
		"employee_column": employee_column,
		"attrition_column": attrition_column,
		"duplicate_count": int(duplicate_count),
		"rows": len(df),
		"columns": len(df.columns),
	}


def clean_dataset(df):
	"""Remove empty rows, empty columns, and duplicate rows."""

	df = df.copy()
	df = df.dropna(how="all")
	df = df.dropna(axis=1, how="all")
	df = df.drop_duplicates()
	df = df.reset_index(drop=True)

	return df


def normalize_target(y):
	"""Convert common attrition values into binary values."""

	y = y.copy()
	y_string = y.astype(str).str.strip().str.lower()

	mapping = {
		"yes": 1,
		"no": 0,
		"y": 1,
		"n": 0,
		"leave": 1,
		"left": 1,
		"leaving": 1,
		"exit": 1,
		"exited": 1,
		"stay": 0,
		"stayed": 0,
		"remain": 0,
		"remained": 0,
		"true": 1,
		"false": 0,
		"1": 1,
		"0": 0,
	}

	normalized = y_string.map(mapping)

	if normalized.isna().any():
		numeric_values = pd.to_numeric(y, errors="coerce")
		normalized = normalized.fillna(numeric_values)

	if normalized.isna().any():
		invalid_values = y[normalized.isna()].unique()
		raise ValueError(
			"Unable to convert target values to binary format. "
			f"Unknown values: {invalid_values}"
		)

	normalized = normalized.astype(int)
	unique_values = set(normalized.unique())

	if not unique_values.issubset({0, 1}):
		raise ValueError(
			"Target column must contain two classes. "
			f"Found values: {unique_values}"
		)

	return normalized


def prepare_features_and_target(df):
	"""Separate employee identifiers, features, and target."""

	df = clean_dataset(df)
	employee_column = find_employee_column(df)
	attrition_column = find_attrition_column(df)

	if employee_column is None:
		raise ValueError("Employee ID or Employee Name column was not found.")

	if attrition_column is None:
		raise ValueError("Attrition/Turnover target column was not found.")

	y = normalize_target(df[attrition_column])
	X = df.drop(columns=[attrition_column])

	if employee_column in X.columns:
		X = X.drop(columns=[employee_column])

	return X, y, employee_column, attrition_column


def build_preprocessor(X):
	"""Create preprocessing pipelines for numerical and categorical features."""

	numerical_columns = X.select_dtypes(
		include=["int64", "int32", "float64", "float32"]
	).columns.tolist()

	categorical_columns = X.select_dtypes(
		include=["object", "category", "bool"]
	).columns.tolist()

	numerical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)

	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			(
				"encoder",
				OneHotEncoder(handle_unknown="ignore", sparse_output=False),
			),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("numerical", numerical_pipeline, numerical_columns),
			("categorical", categorical_pipeline, categorical_columns),
		],
		remainder="drop",
	)

	return preprocessor


def get_feature_information(X):
	"""Return information about numerical and categorical columns."""

	numerical_columns = X.select_dtypes(
		include=["int64", "int32", "float64", "float32"]
	).columns.tolist()

	categorical_columns = X.select_dtypes(
		include=["object", "category", "bool"]
	).columns.tolist()

	return {
		"numerical_features": numerical_columns,
		"categorical_features": categorical_columns,
		"total_features": len(X.columns),
	}
