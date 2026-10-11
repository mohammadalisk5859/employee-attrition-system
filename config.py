from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# PROJECT DIRECTORIES
# ============================================================

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

DATABASE_DIR = BASE_DIR / "database"
MODEL_DIR = BASE_DIR / "models"

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
RAG_DIR = BASE_DIR / "rag"

# ============================================================
# DATABASE
# ============================================================

DATABASE_PATH = DATABASE_DIR / "app.db"

# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILE = MODEL_DIR / "attrition_model.pkl"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.pkl"
MODEL_METADATA_FILE = MODEL_DIR / "model_metadata.json"

# ============================================================
# EMPLOYEE DATA SETTINGS
# ============================================================

# Possible names for employee identification columns
EMPLOYEE_ID_COLUMNS = [
	"EmployeeID",
	"Employee ID",
	"employee_id",
	"employeeid",
	"ID",
	"id"
]

EMPLOYEE_NAME_COLUMNS = [
	"EmployeeName",
	"Employee Name",
	"employee_name",
	"employeename",
	"Name",
	"name"
]

# ============================================================
# ATTRITION TARGET SETTINGS
# ============================================================

ATTRITION_COLUMNS = [
	"Attrition",
	"attrition",
	"Turnover",
	"turnover",
	"LeftCompany",
	"Left Company",
	"left_company",
	"Exited",
	"Exit"
]

# ============================================================
# RISK THRESHOLDS
# ============================================================

# These are initial thresholds.
# We can tune them later based on model evaluation.

LOW_RISK_THRESHOLD = 0.30
MEDIUM_RISK_THRESHOLD = 0.60

# ============================================================
# MODEL SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5

# ============================================================
# OPTIMIZATION SETTINGS
# ============================================================

OPTUNA_TRIALS = 20

# ============================================================
# FILE SETTINGS
# ============================================================

ALLOWED_FILE_TYPES = [
	"csv",
	"xlsx",
	"xls"
]

# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

REQUIRED_DIRECTORIES = [
	DATA_DIR,
	UPLOAD_DIR,
	DATABASE_DIR,
	MODEL_DIR,
	KNOWLEDGE_BASE_DIR,
	RAG_DIR
]

for directory in REQUIRED_DIRECTORIES:
	directory.mkdir(parents=True, exist_ok=True)
