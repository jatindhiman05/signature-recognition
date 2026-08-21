import os
from src.constants import *
from dataclasses import dataclass
from src.utils.main_utils import read_yaml_file


@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        self.BUCKET_NAME: str = self.config['data_ingestion_config']["bucket_name"]
        self.ZIP_FILE_NAME: str = self.config['data_ingestion_config']["zip_file_name"]
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, self.ZIP_FILE_NAME)


class DataTransformationConfig:

    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)

        data_transformation_config = self.config["data_transformation_config"]

        self.STD: list = data_transformation_config["std"]
        self.MEAN: list = data_transformation_config["mean"]
        self.IMG_SIZE: int = data_transformation_config["img_size"]
        self.DEGREE_N: int = data_transformation_config["degree_n"]
        self.DEGREE_P: int = data_transformation_config["degree_p"]
        self.TRAIN_RATIO: float = data_transformation_config["train_ratio"]
        self.VALID_RATIO: float = data_transformation_config["valid_ratio"]

        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(
            os.getcwd(),
            ARTIFACTS_DIR,
            DATA_TRANSFORMATION_ARTIFACTS_DIR
        )

        self.TRAIN_TRANSFORM_OBJECT_FILE_PATH: str = os.path.join(
            self.DATA_TRANSFORMATION_ARTIFACTS_DIR,
            DATA_TRANSFORMATION_TRAIN_FILE_NAME
        )

        self.VALID_TRANSFORM_OBJECT_FILE_PATH: str = os.path.join(
            self.DATA_TRANSFORMATION_ARTIFACTS_DIR,
            DATA_TRANSFORMATION_VALID_FILE_NAME
        )

        self.TEST_TRANSFORM_OBJECT_FILE_PATH: str = os.path.join(
            self.DATA_TRANSFORMATION_ARTIFACTS_DIR,
            DATA_TRANSFORMATION_TEST_FILE_NAME
        )

@dataclass
class ModelTrainerConfig:

    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)

        self.LR: float = self.config["model_trainer_config"]["lr"]
        self.EPOCHS: int = self.config["model_trainer_config"]["epochs"]
        self.NUM_WORKERS: int = self.config["model_trainer_config"]["num_workers"]
        self.BATCH_SIZE: int = self.config["model_trainer_config"]["batch_size"]

        self.MODEL_TRAINER_ARTIFACTS_DIR: str = os.path.join(
            os.getcwd(),
            ARTIFACTS_DIR,
            MODEL_TRAINER_ARTIFACTS_DIR
        )

        self.TRAINED_MODEL_PATH: str = os.path.join(
            self.MODEL_TRAINER_ARTIFACTS_DIR,
            TRAINED_MODEL_PATH
        )


class ModelEvaluationConfig:

    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)

        self.MODEL_NAME: str = MODEL_NAME

        self.BUCKET_NAME: str = self.config[
            "model_evaluation_config"
        ]["bucket_name"]

        self.BATCH_SIZE: int = self.config[
            "model_evaluation_config"
        ]["batch_size"]

        self.NUM_WORKERS: int = self.config[
            "model_evaluation_config"
        ]["num_workers"]

        self.MODEL_EVALUATION_ARTIFACTS_DIR: str = os.path.join(
            os.getcwd(),
            ARTIFACTS_DIR,
            MODEL_EVALUATION_ARTIFACTS_DIR
        )

        self.BEST_MODEL_DIR: str = os.path.join(
            self.MODEL_EVALUATION_ARTIFACTS_DIR,
            BEST_MODEL_DIR
        )

@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        self.MODEL_NAME: str = MODEL_NAME
        self.BUCKET_NAME: str = self.config['model_pusher_config']["bucket_name"]