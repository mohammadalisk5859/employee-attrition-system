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
		match = _find_column(df.columns, column)
		if match is not None:
			return match

	for column in EMPLOYEE_NAME_COLUMNS:
		match = _find_column(df.columns, column)
		if match is not None:
			return match

	return None


def _normalize_column_name(column):
	return "".join(character.lower() for character in str(column) if character.isalnum())


def _find_column(columns, requested_name):
	normalized_name = _normalize_column_name(requested_name)
	return next(
		(column for column in columns if _normalize_column_name(column) == normalized_name),
		None,
	)


def is_employee_identity_column(column):
	"""Return whether a column contains employee identity rather than job attributes."""
	identity_names = EMPLOYEE_ID_COLUMNS + EMPLOYEE_NAME_COLUMNS
	normalized_name = _normalize_column_name(column)
	return normalized_name in {
		_normalize_column_name(name)
		for name in identity_names
	}


def align_features_to_model(X, expected_feature_names, numerical_features=()):
	"""Align current features to an older model while redacting identity inputs."""
	expected = list(expected_feature_names)
	expected_attributes = {
		name for name in expected if not is_employee_identity_column(name)
	}
	actual_attributes = set(X.columns)
	if actual_attributes != expected_attributes:
		raise ValueError("Employee dataset features do not match the saved model schema.")

	aligned = X.copy()
	numerical_features = set(numerical_features)
	for name in expected:
		if is_employee_identity_column(name) and name not in aligned.columns:
			aligned[name] = 0 if name in numerical_features else "__redacted_identity__"
	return aligned.reindex(columns=expected)


def find_attrition_column(df):
	"""Find the attrition/turnover target column."""

	for column in ATTRITION_COLUMNS:
		match = _find_column(df.columns, column)
		if match is not None:
			return match

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
	identity_columns = [
		column for column in X.columns if is_employee_identity_column(column)
	]
	X = X.drop(columns=identity_columns)

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
