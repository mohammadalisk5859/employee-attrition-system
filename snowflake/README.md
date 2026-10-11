# Streamlit in Snowflake deployment

This app is prepared for Streamlit in Snowflake. The repository root contains `environment.yml` for the app environment.

## Before deployment

Choose or create:

- A Snowflake database and schema
- A warehouse with permission to run the Streamlit app
- A stage for the app files

Upload the complete repository, including `app.py`, `config.py`, the `ml`, `rag`, and `explainability` packages, `models`, and `environment.yml`. Do not upload secrets.

## Create the app

In Snowsight:

1. Open **Projects > Streamlit** and select **Create app**.
2. Choose the target database and schema.
3. Choose the warehouse.
4. Upload the repository files, with `app.py` as the main file.
5. Select `environment.yml` as the environment file.
6. Create the app and wait for package resolution.

The app includes persisted model artifacts in `models/`. If those artifacts do not match an uploaded dataset, the app asks the user to train a refreshed model.

## Snowflake runtime notes

- The Web search strategy is optional and may return the offline response unless an External Access Integration is configured for DuckDuckGo.
- Training writes model artifacts to the app runtime filesystem. For durable shared model storage, move model persistence to a Snowflake stage or run training in a separate controlled job.
- Use a warehouse sized for the Optuna/XGBoost training workload; the first model refresh can take several minutes.
- Keep HR documents and employee data inside approved Snowflake governance boundaries.
