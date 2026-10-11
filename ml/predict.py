import json

import joblib

from config import MODEL_FILE, MODEL_METADATA_FILE, PREPROCESSOR_FILE


def load_artifacts():
	"""Load the persisted model, preprocessor, and metadata."""
	missing = [
		path.name
		for path in (MODEL_FILE, PREPROCESSOR_FILE, MODEL_METADATA_FILE)
		if not path.exists()
	]
	if missing:
		raise FileNotFoundError(
			"Model artifacts are missing: " + ", ".join(missing)
		)

	return (
		joblib.load(MODEL_FILE),
		joblib.load(PREPROCESSOR_FILE),
		json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8")),
	)


def predict_attrition(employee_data, model=None, preprocessor=None):
	"""Return attrition probabilities and risk labels for employee records."""
	if model is None or preprocessor is None:
		model, preprocessor, _ = load_artifacts()

	transformed_data = preprocessor.transform(employee_data)
	probabilities = model.predict_proba(transformed_data)[:, 1]

	def risk_label(probability):
		if probability >= 0.60:
			return "High"
		if probability >= 0.30:
			return "Medium"
		return "Low"

	return {
		"probability": probabilities,
		"prediction": (probabilities >= 0.50).astype(int),
		"risk": [risk_label(probability) for probability in probabilities],
	}
