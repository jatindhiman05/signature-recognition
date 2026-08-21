import sys

from src.logger import logging
from src.exception import CustomException

from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifacts,
    ModelPusherArtifacts
)

from src.configurations.gcloud_syncer import GCloudSync


class ModelPusher:

    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_trainer_artifacts: ModelTrainerArtifacts
    ):
        """
        :param model_pusher_config: Configuration for model pusher
        :param model_trainer_artifacts: Output reference of model trainer artifact stage
        """

        self.model_pusher_config = model_pusher_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.gcloud = GCloudSync()

    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        """
        This method initiates the model pusher stage.

        :return: ModelPusherArtifacts
        """

        logging.info(
            "Entered initiate_model_pusher method of ModelPusher class"
        )

        try:

            logging.info("Uploading the model to GCloud storage")

            self.gcloud.sync_file_to_gcloud(
                self.model_pusher_config.BUCKET_NAME,
                self.model_trainer_artifacts.trained_model_path
            )

            logging.info("Uploaded best model to GCloud storage")

            logging.info("Saving the model pusher artifacts")

            model_pusher_artifact = ModelPusherArtifacts(
                bucket_name=self.model_pusher_config.BUCKET_NAME
            )

            logging.info(
                "Exited the initiate_model_pusher method of ModelPusher class"
            )

            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e