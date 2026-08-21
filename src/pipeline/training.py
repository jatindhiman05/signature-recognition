import sys

from src.logger import logging
from src.exception import CustomException

from src.components.model_pusher import ModelPusher
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

from src.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
    ModelTrainerArtifacts,
    ModelEvaluationArtifacts,
    ModelPusherArtifacts
)


class TrainingPipeline:

    def __init__(self):

        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    # ============================================================
    # DATA INGESTION
    # ============================================================

    def start_data_ingestion(self) -> DataIngestionArtifacts:

        logging.info(
            "Entered the start_data_ingestion method "
            "of TrainingPipeline class"
        )

        try:

            logging.info(
                "Getting the dataset from S3 Storage bucket"
            )

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifacts = (
                data_ingestion.initiate_data_ingestion()
            )

            logging.info(
                "Got the dataset from S3 Storage"
            )

            logging.info(
                "Exited the start_data_ingestion method "
                "of TrainingPipeline class"
            )

            return data_ingestion_artifacts

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # DATA TRANSFORMATION
    # ============================================================

    def start_data_transformation(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts
    ) -> DataTransformationArtifacts:

        logging.info(
            "Entered the start_data_transformation method "
            "of TrainingPipeline class"
        )

        try:

            data_transformation = DataTransformation(
                data_ingestion_artifacts=data_ingestion_artifacts,
                data_transformation_config=self.data_transformation_config
            )

            data_transformation_artifacts = (
                data_transformation.initiate_data_transformation()
            )

            logging.info(
                "Data transformation completed"
            )

            logging.info(
                "Exited the start_data_transformation method "
                "of TrainingPipeline class"
            )

            return data_transformation_artifacts

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # MODEL TRAINING
    # ============================================================

    def start_model_trainer(
        self,
        data_transformation_artifacts: DataTransformationArtifacts
    ) -> ModelTrainerArtifacts:

        logging.info(
            "Entered the start_model_trainer method "
            "of TrainingPipeline class"
        )

        try:

            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifacts=data_transformation_artifacts
            )

            model_trainer_artifacts = (
                model_trainer.initiate_model_trainer()
            )

            logging.info(
                "Model training completed"
            )

            logging.info(
                "Exited the start_model_trainer method "
                "of TrainingPipeline class"
            )

            return model_trainer_artifacts

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # MODEL EVALUATION
    # ============================================================

    def start_model_evaluation(
        self,
        model_trainer_artifacts: ModelTrainerArtifacts,
        data_transformation_artifacts: DataTransformationArtifacts
    ) -> ModelEvaluationArtifacts:

        logging.info(
            "Entered the start_model_evaluation method "
            "of TrainingPipeline class"
        )

        try:

            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                data_transformation_artifacts=data_transformation_artifacts,
                model_trainer_artifacts=model_trainer_artifacts
            )

            model_evaluation_artifacts = (
                model_evaluation.initiate_model_evaluation()
            )

            logging.info(
                "Model evaluation completed"
            )

            logging.info(
                "Exited the start_model_evaluation method "
                "of TrainingPipeline class"
            )

            return model_evaluation_artifacts

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # MODEL PUSHER
    # ============================================================

    def start_model_pusher(
        self,
        model_trainer_artifacts: ModelTrainerArtifacts
    ) -> ModelPusherArtifacts:

        logging.info(
            "Entered the start_model_pusher method "
            "of TrainingPipeline class"
        )

        try:

            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_trainer_artifacts=model_trainer_artifacts
            )

            model_pusher_artifact = (
                model_pusher.initiate_model_pusher()
            )

            logging.info(
                "Model pushing completed"
            )

            logging.info(
                "Exited the start_model_pusher method "
                "of TrainingPipeline class"
            )

            return model_pusher_artifact

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # RUN COMPLETE TRAINING PIPELINE
    # ============================================================

    def run_pipeline(self) -> None:

        logging.info(
            "Entered the run_pipeline method "
            "of TrainingPipeline class"
        )

        try:

            # ----------------------------------------------------
            # 1. DATA INGESTION
            # ----------------------------------------------------

            data_ingestion_artifacts = (
                self.start_data_ingestion()
            )

            logging.info(
                f"Data ingestion completed: "
                f"{data_ingestion_artifacts}"
            )

            # ----------------------------------------------------
            # 2. DATA TRANSFORMATION
            # ----------------------------------------------------

            data_transformation_artifacts = (
                self.start_data_transformation(
                    data_ingestion_artifacts=data_ingestion_artifacts
                )
            )

            logging.info(
                f"Data transformation completed: "
                f"{data_transformation_artifacts}"
            )

            # ----------------------------------------------------
            # 3. MODEL TRAINING
            # ----------------------------------------------------

            model_trainer_artifacts = (
                self.start_model_trainer(
                    data_transformation_artifacts=
                    data_transformation_artifacts
                )
            )

            logging.info(
                f"Model training completed: "
                f"{model_trainer_artifacts}"
            )

            # ----------------------------------------------------
            # 4. MODEL EVALUATION
            # ----------------------------------------------------

            model_evaluation_artifacts = (
                self.start_model_evaluation(
                    model_trainer_artifacts=model_trainer_artifacts,
                    data_transformation_artifacts=
                    data_transformation_artifacts
                )
            )

            logging.info(
                f"Model evaluation completed: "
                f"{model_evaluation_artifacts}"
            )

            # ----------------------------------------------------
            # 5. MODEL PUSHER
            # ----------------------------------------------------

            if not model_evaluation_artifacts.is_model_accepted:
                raise Exception("Training model is not better than the best model.")

            model_pusher_artifacts = (
                self.start_model_pusher(
                    model_trainer_artifacts=model_trainer_artifacts
                )
            )

            logging.info(
                f"Model pusher completed: "
                f"{model_pusher_artifacts}"
            )

            # ----------------------------------------------------
            # PIPELINE COMPLETED
            # ----------------------------------------------------

            logging.info(
                "Training pipeline completed successfully"
            )

        except Exception as e:

            raise CustomException(e, sys) from e