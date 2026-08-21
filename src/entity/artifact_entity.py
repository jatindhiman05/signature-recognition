from dataclasses import dataclass, asdict

# Data Ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    dataset_path: str

    def to_dict(self):
        return self.__dict__

@dataclass
class DataTransformationArtifacts:
    train_transformed_object: str
    valid_transformed_object: str
    test_transformed_object: str
    classes: int

    def to_dict(self):
        return self.__dict__

# Model trainer artifacts
@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str

    def to_dict(self):
        return self.__dict__

# Model Evaluation artifacts
@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool

    def to_dict(self):
        return self.__dict__

@dataclass
class ModelPusherArtifacts:
    bucket_name: str

    def to_dict(self):
        return self.__dict__