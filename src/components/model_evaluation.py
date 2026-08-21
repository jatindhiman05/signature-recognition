import os
import sys

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.logger import logging
from src.exception import CustomException

from src.entity.config_entity import ModelEvaluationConfig

from src.entity.artifact_entity import (
    ModelTrainerArtifacts,
    DataTransformationArtifacts,
    ModelEvaluationArtifacts
)

from src.configurations.gcloud_syncer import GCloudSync
from src.constants import DEVICE
from src.utils.main_utils import load_object


class ModelEvaluation:

    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifacts: ModelTrainerArtifacts,
        data_transformation_artifacts: DataTransformationArtifacts
    ):
        """
        :param model_evaluation_config:
            Configuration for model evaluation

        :param model_trainer_artifacts:
            Output reference of model trainer artifact stage

        :param data_transformation_artifacts:
            Output reference of data transformation artifact stage
        """

        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts

        self.gcloud = GCloudSync()

    # ============================================================
    # CHECK WHETHER MODEL EXISTS IN GCLOUD
    # ============================================================

    def is_best_model_available(self) -> bool:
        """
        Check whether a previous best model exists
        in GCloud Storage.

        Returns:
            True  -> model exists
            False -> model does not exist
        """

        try:

            logging.info(
                "Checking whether previous best model exists "
                "in GCloud Storage"
            )

            bucket_name = (
                self.model_evaluation_config.BUCKET_NAME
            )

            model_name = (
                self.model_evaluation_config.MODEL_NAME
            )

            model_exists = (
                self.gcloud.is_file_exist_in_gcloud(
                    gcp_bucket_url=bucket_name,
                    filename=model_name
                )
            )

            if model_exists:

                logging.info(
                    f"Previous best model found in GCloud: "
                    f"gs://{bucket_name}/{model_name}"
                )

            else:

                logging.info(
                    f"No previous best model found in GCloud: "
                    f"gs://{bucket_name}/{model_name}"
                )

            return model_exists

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # GET BEST MODEL FROM GCLOUD
    # ============================================================

    def get_best_model_from_gcloud(self) -> str:
        """
        Download the previous best model from GCloud Storage.

        Returns:
            Local path of downloaded model.
        """

        try:

            logging.info(
                "Entered get_best_model_from_gcloud method"
            )

            os.makedirs(
                self.model_evaluation_config.BEST_MODEL_DIR,
                exist_ok=True
            )

            self.gcloud.sync_file_from_gcloud(
                self.model_evaluation_config.BUCKET_NAME,
                self.model_evaluation_config.MODEL_NAME,
                self.model_evaluation_config.BEST_MODEL_DIR
            )

            best_model_path = os.path.join(
                self.model_evaluation_config.BEST_MODEL_DIR,
                self.model_evaluation_config.MODEL_NAME
            )

            logging.info(
                f"Previous best model downloaded to: "
                f"{best_model_path}"
            )

            logging.info(
                "Exited get_best_model_from_gcloud method"
            )

            return best_model_path

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # EVALUATE MODEL
    # ============================================================

    def evaluate(
        self,
        model,
        criterion,
        test_dataloader
    ):
        """
        Evaluate the given model on the test dataset.

        Returns:
            Average test loss.
        """

        try:

            logging.info(
                "Entered evaluate method"
            )

            total_test_loss = 0.0

            model.eval()

            with torch.no_grad():

                with tqdm(
                    test_dataloader,
                    unit="batch",
                    leave=False
                ) as pbar:

                    pbar.set_description("Testing")

                    for images, idxs in pbar:

                        images = images.to(
                            DEVICE,
                            non_blocking=True
                        )

                        idxs = idxs.to(
                            DEVICE,
                            non_blocking=True
                        )

                        output = model(images)

                        loss = criterion(
                            output,
                            idxs
                        )

                        total_test_loss += loss.item()

            test_loss = (
                total_test_loss /
                len(test_dataloader)
            )

            logging.info(
                f"Test loss: {test_loss:.4f}"
            )

            logging.info(
                "Exited evaluate method"
            )

            return test_loss

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # INITIATE MODEL EVALUATION
    # ============================================================

    def initiate_model_evaluation(
        self
    ) -> ModelEvaluationArtifacts:

        """
        Initiate all steps of model evaluation.

        Logic:

        1. Evaluate the newly trained model.
        2. Check whether a previous model exists in GCloud.
        3. If no previous model exists:
               accept the newly trained model.
        4. If previous model exists:
               download it,
               evaluate it,
               compare losses.
        5. Accept new model only if it performs better.
        """

        logging.info(
            "Entered initiate_model_evaluation method"
        )

        try:

            # ====================================================
            # 1. LOAD TEST DATASET
            # ====================================================

            logging.info(
                "Loading test dataset"
            )

            test_dataset = load_object(
                self.data_transformation_artifacts
                .test_transformed_object
            )

            test_loader = DataLoader(
                test_dataset,
                shuffle=False,
                batch_size=(
                    self.model_evaluation_config.BATCH_SIZE
                ),
                num_workers=(
                    self.model_evaluation_config.NUM_WORKERS
                )
            )

            criterion = torch.nn.CrossEntropyLoss()

            # ====================================================
            # 2. LOAD CURRENTLY TRAINED MODEL
            # ====================================================

            logging.info(
                "Loading currently trained model"
            )

            trained_model = torch.load(
                self.model_trainer_artifacts.trained_model_path,
                map_location=DEVICE,
                weights_only=False
            )

            trained_model.eval()

            # ====================================================
            # 3. EVALUATE CURRENTLY TRAINED MODEL
            # ====================================================

            logging.info(
                "Evaluating currently trained model"
            )

            trained_model_loss = self.evaluate(
                model=trained_model,
                criterion=criterion,
                test_dataloader=test_loader
            )

            logging.info(
                f"Currently trained model loss: "
                f"{trained_model_loss:.4f}"
            )

            # ====================================================
            # 4. CHECK WHETHER PREVIOUS MODEL EXISTS
            # ====================================================

            logging.info(
                "Checking whether previous best model exists"
            )

            best_model_exists = (
                self.is_best_model_available()
            )

            # ====================================================
            # 5. FIRST TRAINING RUN
            # ====================================================

            if not best_model_exists:

                logging.info(
                    "No previous best model found."
                )

                logging.info(
                    "This is the first training run."
                )

                logging.info(
                    "Automatically accepting "
                    "the currently trained model."
                )

                is_model_accepted = True

            # ====================================================
            # 6. PREVIOUS MODEL EXISTS
            # ====================================================

            else:

                logging.info(
                    "Previous best model found."
                )

                logging.info(
                    "Downloading previous best model "
                    "for comparison."
                )

                best_model_path = (
                    self.get_best_model_from_gcloud()
                )

                # =================================================
                # LOAD PREVIOUS BEST MODEL
                # =================================================

                logging.info(
                    "Loading previous best model"
                )

                best_model = torch.load(
                    best_model_path,
                    map_location=DEVICE,
                    weights_only=False
                )

                best_model.eval()

                # =================================================
                # EVALUATE PREVIOUS BEST MODEL
                # =================================================

                logging.info(
                    "Evaluating previous best model"
                )

                best_model_loss = self.evaluate(
                    model=best_model,
                    criterion=criterion,
                    test_dataloader=test_loader
                )

                logging.info(
                    f"Previous best model loss: "
                    f"{best_model_loss:.4f}"
                )

                # =================================================
                # COMPARE BOTH MODELS
                # =================================================

                logging.info(
                    "Comparing currently trained model "
                    "with previous best model"
                )

                if trained_model_loss < best_model_loss:

                    is_model_accepted = True

                    logging.info(
                        f"Currently trained model loss "
                        f"({trained_model_loss:.4f}) is lower than "
                        f"previous best model loss "
                        f"({best_model_loss:.4f})."
                    )

                    logging.info(
                        "Currently trained model accepted."
                    )

                else:

                    is_model_accepted = False

                    logging.info(
                        f"Currently trained model loss "
                        f"({trained_model_loss:.4f}) is not lower "
                        f"than previous best model loss "
                        f"({best_model_loss:.4f})."
                    )

                    logging.info(
                        "Currently trained model rejected."
                    )

            # ====================================================
            # 7. CREATE MODEL EVALUATION ARTIFACT
            # ====================================================

            model_evaluation_artifacts = (
                ModelEvaluationArtifacts(
                    is_model_accepted=is_model_accepted
                )
            )

            logging.info(
                f"Model evaluation completed. "
                f"Model accepted: {is_model_accepted}"
            )

            logging.info(
                "Exited initiate_model_evaluation method"
            )

            return model_evaluation_artifacts

        except Exception as e:

            raise CustomException(e, sys) from e