from dataclasses import dataclass
from pathlib import Path


# Data ingestion is fine. 
@dataclass(frozen=True) # frozen means it is immutable. 
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict

### I have to make changes here. 
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    test_size: float
    target_column: str


# I have to make changes here to match the right parameters. 
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    learning_rate: float
    n_estimators: int
    max_depth: int
    model_name: str
    target_column: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str