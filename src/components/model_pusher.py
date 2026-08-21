import sys

from src.logger import logging
from src.exception import CustomException

from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifacts,
    ModelPusherArtifacts
)

from src.configurations.s3_syncer import S3Sync


class ModelPusher:

    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_trainer_artifacts: ModelTrainerArtifacts
    ):
        """
        :param model_pusher_config:
            Configuration for model pusher.

        :param model_trainer_artifacts:
            Output reference of model trainer artifact stage.
        """

        self.model_pusher_config = model_pusher_config

        self.model_trainer_artifacts = (
            model_trainer_artifacts
        )

        # AWS S3 storage handler
        self.s3 = S3Sync()

    # ============================================================
    # PUSH MODEL TO S3
    # ============================================================

    def initiate_model_pusher(
        self
    ) -> ModelPusherArtifacts:

        """
        Upload the trained model to AWS S3.

        Returns:
            ModelPusherArtifacts
        """

        logging.info(
            "Entered initiate_model_pusher method "
            "of ModelPusher class"
        )

        try:

            # ----------------------------------------------------
            # MODEL INFORMATION
            # ----------------------------------------------------

            bucket_name = (
                self.model_pusher_config.BUCKET_NAME
            )

            model_path = (
                self.model_trainer_artifacts
                .trained_model_path
            )

            logging.info(
                "Uploading trained model to AWS S3"
            )

            logging.info(
                f"Local model path: {model_path}"
            )

            logging.info(
                f"S3 bucket: {bucket_name}"
            )

            # ----------------------------------------------------
            # UPLOAD MODEL
            # ----------------------------------------------------

            self.s3.sync_file_to_s3(
                bucket_name=bucket_name,
                filepath=model_path
            )

            logging.info(
                "Successfully uploaded trained model "
                "to AWS S3"
            )

            # ----------------------------------------------------
            # CREATE ARTIFACT
            # ----------------------------------------------------

            model_pusher_artifact = (
                ModelPusherArtifacts(
                    bucket_name=bucket_name
                )
            )

            logging.info(
                f"Model pusher artifact: "
                f"{model_pusher_artifact}"
            )

            logging.info(
                "Exited initiate_model_pusher method "
                "of ModelPusher class"
            )

            return model_pusher_artifact

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e