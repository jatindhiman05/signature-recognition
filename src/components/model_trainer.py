import os
import sys

import torch
import torch.nn as nn

from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader

from src.logger import logging
from src.constants import DEVICE
from src.exception import CustomException
from src.utils.main_utils import load_object

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifacts,
    ModelTrainerArtifacts
)


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifacts: DataTransformationArtifacts
    ):

        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifacts = (
            data_transformation_artifacts
        )

        self.learning_rate = (
            self.model_trainer_config.LR
        )

        self.epochs = (
            self.model_trainer_config.EPOCHS
        )

        self.batch_size = (
            self.model_trainer_config.BATCH_SIZE
        )

        self.num_workers = (
            self.model_trainer_config.NUM_WORKERS
        )

    # ============================================================
    # TRAIN + VALIDATE FOR ONE EPOCH
    # ============================================================

    def train_one_epoch(
        self,
        model,
        criterion,
        optimizer,
        train_dataloader,
        valid_dataloader
    ):

        try:

            # ====================================================
            # TRAINING
            # ====================================================

            model.train()

            total_train_loss = 0.0

            with tqdm(
                train_dataloader,
                unit="batch",
                leave=False
            ) as pbar:

                pbar.set_description("Training")

                for images, idxs in pbar:

                    images = images.to(
                        DEVICE,
                        non_blocking=True
                    )

                    idxs = idxs.to(
                        DEVICE,
                        non_blocking=True
                    )

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    output = model(images)

                    loss = criterion(
                        output,
                        idxs
                    )

                    loss.backward()

                    optimizer.step()

                    total_train_loss += loss.item()

            train_loss = (
                total_train_loss /
                len(train_dataloader)
            )

            # ====================================================
            # VALIDATION
            # ====================================================

            model.eval()

            total_valid_loss = 0.0

            with torch.no_grad():

                with tqdm(
                    valid_dataloader,
                    unit="batch",
                    leave=False
                ) as pbar:

                    pbar.set_description("Validation")

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

                        total_valid_loss += loss.item()

            valid_loss = (
                total_valid_loss /
                len(valid_dataloader)
            )

            return train_loss, valid_loss

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # INITIATE MODEL TRAINER
    # ============================================================

    def initiate_model_trainer(
        self
    ) -> ModelTrainerArtifacts:

        try:

            logging.info(
                "Entered initiate_model_trainer method"
            )

            # ====================================================
            # LOAD DATASETS
            # ====================================================

            train_dataset = load_object(
                self.data_transformation_artifacts
                .train_transformed_object
            )

            valid_dataset = load_object(
                self.data_transformation_artifacts
                .valid_transformed_object
            )

            logging.info(
                "Loaded train and validation datasets"
            )

            # ====================================================
            # DATALOADERS
            # ====================================================

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )

            logging.info(
                "Created train and validation dataloaders"
            )

            # ====================================================
            # MODEL
            # ====================================================

            model = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT
            )

            # ----------------------------------------------------
            # Convert pretrained RGB conv1 → grayscale conv1
            # ----------------------------------------------------

            old_conv = model.conv1

            model.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

            # Preserve pretrained RGB information
            with torch.no_grad():

                model.conv1.weight.copy_(
                    old_conv.weight.mean(
                        dim=1,
                        keepdim=True
                    )
                )

            # ----------------------------------------------------
            # Replace classifier
            # ----------------------------------------------------

            model.fc = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(
                    model.fc.in_features,
                    self.data_transformation_artifacts.classes
                )
            )

            model = model.to(DEVICE)

            logging.info(
                "ResNet34 model created with 1-channel input"
            )

            # ====================================================
            # LOSS
            # ====================================================

            criterion = nn.CrossEntropyLoss()

            # ====================================================
            # OPTIMIZER
            # ====================================================

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=0.9
            )

            # ====================================================
            # BEST MODEL TRACKING
            # ====================================================

            best_valid_loss = float("inf")

            best_model_state = None

            # ====================================================
            # TRAINING LOOP
            # ====================================================

            logging.info(
                "Model training started"
            )

            for epoch in range(self.epochs):

                print(
                    f"\nEpoch {epoch + 1}/{self.epochs}"
                )

                logging.info(
                    f"Starting epoch {epoch + 1}"
                )

                train_loss, valid_loss = (
                    self.train_one_epoch(
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        train_dataloader=train_loader,
                        valid_dataloader=valid_loader
                    )
                )

                print(
                    f"Train loss: {train_loss:.4f} "
                    f"Valid loss: {valid_loss:.4f}"
                )

                logging.info(
                    f"Epoch {epoch + 1} | "
                    f"Train loss: {train_loss:.4f} | "
                    f"Valid loss: {valid_loss:.4f}"
                )

                # =================================================
                # SAVE BEST MODEL BASED ON VALIDATION LOSS
                # =================================================

                if valid_loss < best_valid_loss:

                    best_valid_loss = valid_loss

                    best_model_state = {
                        key: value.detach().cpu().clone()
                        for key, value
                        in model.state_dict().items()
                    }

                    logging.info(
                        f"New best model found at epoch "
                        f"{epoch + 1} with validation loss "
                        f"{valid_loss:.4f}"
                    )

            # ====================================================
            # LOAD BEST MODEL
            # ====================================================

            if best_model_state is None:

                raise RuntimeError(
                    "Best model was not created."
                )

            model.load_state_dict(
                best_model_state
            )

            model = model.to(DEVICE)

            logging.info(
                f"Best validation loss: "
                f"{best_valid_loss:.4f}"
            )

            # ====================================================
            # SAVE BEST MODEL
            # ====================================================

            os.makedirs(
                self.model_trainer_config
                .MODEL_TRAINER_ARTIFACTS_DIR,
                exist_ok=True
            )

            torch.save(
                model,
                self.model_trainer_config
                .TRAINED_MODEL_PATH
            )

            logging.info(
                f"Best model saved at: "
                f"{self.model_trainer_config.TRAINED_MODEL_PATH}"
            )

            # ====================================================
            # CREATE ARTIFACT
            # ====================================================

            model_trainer_artifacts = (
                ModelTrainerArtifacts(
                    trained_model_path=
                    self.model_trainer_config
                    .TRAINED_MODEL_PATH
                )
            )

            logging.info(
                f"Model trainer artifact: "
                f"{model_trainer_artifacts}"
            )

            logging.info(
                "Exited initiate_model_trainer method"
            )

            return model_trainer_artifacts

        except Exception as e:

            raise CustomException(e, sys) from e