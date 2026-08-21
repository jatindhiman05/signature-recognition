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

from src.configurations.s3_syncer import S3Sync
from src.constants import DEVICE
from src.utils.main_utils import load_object


class ModelEvaluation:

    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifacts: ModelTrainerArtifacts,
        data_transformation_artifacts: DataTransformationArtifacts
    ):

        self.model_evaluation_config = (
            model_evaluation_config
        )

        self.model_trainer_artifacts = (
            model_trainer_artifacts
        )

        self.data_transformation_artifacts = (
            data_transformation_artifacts
        )

        # ========================================================
        # AWS S3
        # ========================================================

        self.s3 = S3Sync()

    # ============================================================
    # CHECK WHETHER PREVIOUS BEST MODEL EXISTS
    # ============================================================

    def is_best_model_available(self) -> bool:

        try:

            logging.info(
                "Checking whether previous best model exists "
                "in AWS S3"
            )

            bucket_name = (
                self.model_evaluation_config.BUCKET_NAME
            )

            model_name = (
                self.model_evaluation_config.MODEL_NAME
            )

            model_exists = (
                self.s3.is_file_exist_in_s3(
                    bucket_name=bucket_name,
                    filename=model_name
                )
            )

            if model_exists:

                logging.info(
                    f"Previous best model found: "
                    f"s3://{bucket_name}/{model_name}"
                )

            else:

                logging.info(
                    f"No previous best model found: "
                    f"s3://{bucket_name}/{model_name}"
                )

            return model_exists

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ============================================================
    # DOWNLOAD PREVIOUS BEST MODEL
    # ============================================================

    def get_best_model_from_s3(self) -> str:

        try:

            logging.info(
                "Downloading previous best model from AWS S3"
            )

            best_model_dir = (
                self.model_evaluation_config.BEST_MODEL_DIR
            )

            os.makedirs(
                best_model_dir,
                exist_ok=True
            )

            best_model_path = os.path.join(
                best_model_dir,
                self.model_evaluation_config.MODEL_NAME
            )

            # ----------------------------------------------------
            # DOWNLOAD TO ACTUAL FILE PATH
            # ----------------------------------------------------

            self.s3.sync_file_from_s3(
                bucket_name=(
                    self.model_evaluation_config.BUCKET_NAME
                ),
                filename=(
                    self.model_evaluation_config.MODEL_NAME
                ),
                destination=best_model_path
            )

            logging.info(
                f"Previous best model downloaded to: "
                f"{best_model_path}"
            )

            return best_model_path

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ============================================================
    # EVALUATE MODEL
    # ============================================================

    def evaluate(
        self,
        model,
        criterion,
        dataloader,
        phase="Validation"
    ):

        try:

            model.eval()

            total_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():

                with tqdm(
                    dataloader,
                    unit="batch",
                    leave=False
                ) as pbar:

                    pbar.set_description(phase)

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

                        total_loss += loss.item()

                        predictions = torch.argmax(
                            output,
                            dim=1
                        )

                        correct += (
                            predictions == idxs
                        ).sum().item()

                        total += idxs.size(0)

            average_loss = (
                total_loss /
                len(dataloader)
            )

            accuracy = (
                correct / total
                if total > 0
                else 0.0
            )

            logging.info(
                f"{phase} loss: "
                f"{average_loss:.4f}"
            )

            logging.info(
                f"{phase} accuracy: "
                f"{accuracy:.4f}"
            )

            return average_loss, accuracy

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ============================================================
    # INITIATE MODEL EVALUATION
    # ============================================================

    def initiate_model_evaluation(
        self
    ) -> ModelEvaluationArtifacts:

        try:

            logging.info(
                "Entered initiate_model_evaluation method"
            )

            criterion = torch.nn.CrossEntropyLoss()

            # ====================================================
            # 1. LOAD VALIDATION DATASET
            # ====================================================

            logging.info(
                "Loading validation dataset"
            )

            valid_dataset = load_object(
                self.data_transformation_artifacts
                .valid_transformed_object
            )

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=(
                    self.model_evaluation_config.BATCH_SIZE
                ),
                shuffle=False,
                num_workers=(
                    self.model_evaluation_config.NUM_WORKERS
                )
            )

            logging.info(
                "Validation dataset loaded"
            )

            # ====================================================
            # 2. LOAD NEWLY TRAINED MODEL
            # ====================================================

            logging.info(
                "Loading newly trained model"
            )

            trained_model = torch.load(
                self.model_trainer_artifacts
                .trained_model_path,
                map_location=DEVICE,
                weights_only=False
            )

            trained_model = trained_model.to(DEVICE)

            # ====================================================
            # 3. EVALUATE NEW MODEL ON VALIDATION SET
            # ====================================================

            logging.info(
                "Evaluating newly trained model "
                "on validation dataset"
            )

            trained_valid_loss, trained_valid_accuracy = (
                self.evaluate(
                    model=trained_model,
                    criterion=criterion,
                    dataloader=valid_loader,
                    phase="Validation - New Model"
                )
            )

            logging.info(
                f"New model validation loss: "
                f"{trained_valid_loss:.4f}"
            )

            logging.info(
                f"New model validation accuracy: "
                f"{trained_valid_accuracy:.4f}"
            )

            # ====================================================
            # 4. CHECK PREVIOUS BEST MODEL
            # ====================================================

            best_model_exists = (
                self.is_best_model_available()
            )

            # ====================================================
            # 5. FIRST TRAINING RUN
            # ====================================================

            if not best_model_exists:

                logging.info(
                    "No previous best model exists."
                )

                logging.info(
                    "First training run detected."
                )

                is_model_accepted = True

            # ====================================================
            # 6. COMPARE WITH PREVIOUS BEST MODEL
            # ====================================================

            else:

                logging.info(
                    "Previous best model exists."
                )

                best_model_path = (
                    self.get_best_model_from_s3()
                )

                logging.info(
                    "Loading previous best model"
                )

                best_model = torch.load(
                    best_model_path,
                    map_location=DEVICE,
                    weights_only=False
                )

                best_model = best_model.to(DEVICE)

                # ------------------------------------------------
                # EVALUATE OLD MODEL ON VALIDATION SET
                # ------------------------------------------------

                logging.info(
                    "Evaluating previous best model "
                    "on validation dataset"
                )

                best_valid_loss, best_valid_accuracy = (
                    self.evaluate(
                        model=best_model,
                        criterion=criterion,
                        dataloader=valid_loader,
                        phase="Validation - Previous Model"
                    )
                )

                logging.info(
                    f"Previous model validation loss: "
                    f"{best_valid_loss:.4f}"
                )

                logging.info(
                    f"Previous model validation accuracy: "
                    f"{best_valid_accuracy:.4f}"
                )

                # =================================================
                # COMPARE VALIDATION LOSSES
                # =================================================

                logging.info(
                    "Comparing new model and previous model "
                    "using validation loss"
                )

                if trained_valid_loss < best_valid_loss:

                    is_model_accepted = True

                    logging.info(
                        f"New model validation loss "
                        f"({trained_valid_loss:.4f}) is lower than "
                        f"previous model validation loss "
                        f"({best_valid_loss:.4f})."
                    )

                    logging.info(
                        "New model ACCEPTED."
                    )

                else:

                    is_model_accepted = False

                    logging.info(
                        f"New model validation loss "
                        f"({trained_valid_loss:.4f}) is NOT lower than "
                        f"previous model validation loss "
                        f"({best_valid_loss:.4f})."
                    )

                    logging.info(
                        "New model REJECTED."
                    )

            # ====================================================
            # 7. FINAL TEST
            # ====================================================

            if is_model_accepted:

                logging.info(
                    "Accepted model will now be evaluated "
                    "on the TEST dataset."
                )

                test_dataset = load_object(
                    self.data_transformation_artifacts
                    .test_transformed_object
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=(
                        self.model_evaluation_config.BATCH_SIZE
                    ),
                    shuffle=False,
                    num_workers=(
                        self.model_evaluation_config.NUM_WORKERS
                    )
                )

                test_loss, test_accuracy = (
                    self.evaluate(
                        model=trained_model,
                        criterion=criterion,
                        dataloader=test_loader,
                        phase="Final Test"
                    )
                )

                logging.info(
                    f"FINAL TEST LOSS: "
                    f"{test_loss:.4f}"
                )

                logging.info(
                    f"FINAL TEST ACCURACY: "
                    f"{test_accuracy:.4f}"
                )

                print(
                    f"\nFinal Test Loss: "
                    f"{test_loss:.4f}"
                )

                print(
                    f"Final Test Accuracy: "
                    f"{test_accuracy * 100:.2f}%"
                )

            else:

                logging.info(
                    "New model rejected."
                )

                logging.info(
                    "Test dataset will NOT be used."
                )

            # ====================================================
            # 8. CREATE ARTIFACT
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

            raise CustomException(
                e,
                sys
            ) from e