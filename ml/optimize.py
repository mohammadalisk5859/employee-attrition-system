import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

from config import RANDOM_STATE, CV_FOLDS, OPTUNA_TRIALS


def optimize_random_forest(X, y, n_trials=None):
	"""Optimize Random Forest hyperparameters using Optuna."""

	if n_trials is None:
		n_trials = OPTUNA_TRIALS

	def objective(trial):
		params = {
			"n_estimators": trial.suggest_int("n_estimators", 100, 500),
			"max_depth": trial.suggest_int("max_depth", 3, 20),
			"min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
			"min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
			"max_features": trial.suggest_categorical(
				"max_features", ["sqrt", "log2", None]
			),
			"class_weight": trial.suggest_categorical(
				"class_weight", ["balanced", "balanced_subsample", None]
			),
			"random_state": RANDOM_STATE,
			"n_jobs": -1,
		}

		model = RandomForestClassifier(**params)
		scores = cross_val_score(
			model,
			X,
			y,
			cv=CV_FOLDS,
			scoring="f1",
			n_jobs=-1,
		)

		return scores.mean()

	study = optuna.create_study(
		direction="maximize",
		study_name="random_forest_optimization",
	)
	study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

	best_params = study.best_params
	best_params["random_state"] = RANDOM_STATE
	best_params["n_jobs"] = -1

	best_model = RandomForestClassifier(**best_params)
	best_model.fit(X, y)

	return best_model, best_params


def optimize_xgboost(X, y, n_trials=None):
	"""Optimize XGBoost hyperparameters using Optuna."""

	if n_trials is None:
		n_trials = OPTUNA_TRIALS

	def objective(trial):
		params = {
			"n_estimators": trial.suggest_int("n_estimators", 100, 500),
			"max_depth": trial.suggest_int("max_depth", 3, 10),
			"learning_rate": trial.suggest_float(
				"learning_rate", 0.01, 0.3, log=True
			),
			"subsample": trial.suggest_float("subsample", 0.6, 1.0),
			"colsample_bytree": trial.suggest_float(
				"colsample_bytree", 0.6, 1.0
			),
			"min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
			"gamma": trial.suggest_float("gamma", 0.0, 5.0),
			"reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
			"reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
			"random_state": RANDOM_STATE,
			"eval_metric": "logloss",
			"n_jobs": -1,
		}

		model = XGBClassifier(**params)
		scores = cross_val_score(
			model,
			X,
			y,
			cv=CV_FOLDS,
			scoring="f1",
			n_jobs=-1,
		)

		return scores.mean()

	study = optuna.create_study(
		direction="maximize",
		study_name="xgboost_optimization",
	)
	study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

	best_params = study.best_params
	best_params["random_state"] = RANDOM_STATE
	best_params["eval_metric"] = "logloss"
	best_params["n_jobs"] = -1

	best_model = XGBClassifier(**best_params)
	best_model.fit(X, y)

	return best_model, best_params


def optimize_models(X, y, n_trials=None):
	"""Optimize both Random Forest and XGBoost."""

	print("\nStarting Random Forest optimization...")
	rf_model, rf_params = optimize_random_forest(X, y, n_trials)
	print("\nRandom Forest optimization completed.")

	print("\nStarting XGBoost optimization...")
	xgb_model, xgb_params = optimize_xgboost(X, y, n_trials)
	print("\nXGBoost optimization completed.")

	return {
		"random_forest": {
			"model": rf_model,
			"params": rf_params,
		},
		"xgboost": {
			"model": xgb_model,
			"params": xgb_params,
		},
	}
