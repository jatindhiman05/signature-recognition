import os
import sys
import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
from src.logger import logging
from src.constants import DEVICE
from torch.utils.data import DataLoader
from src.exception import CustomException
from src.utils.main_utils import load_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifacts,ModelTrainerArtifacts

class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifacts: DataTransformationArtifacts
    ):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifacts = data_transformation_artifacts

        self.learning_rate = self.model_trainer_config.LR
        self.epochs = self.model_trainer_config.EPOCHS
        self.batch_size = self.model_trainer_config.BATCH_SIZE
        self.num_workers = self.model_trainer_config.NUM_WORKERS

    def train(
        self,
        model,
        criterion,
        optimizer,
        train_dataloader,
        valid_dataloader
    ):

        try:
            total_train_loss = 0
            total_valid_loss = 0

            # ---------------- TRAINING ----------------

            model.train()

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

                    output = model(images)

                    loss = criterion(output, idxs)

                    total_train_loss += loss.item()

                    loss.backward()

                    optimizer.step()

                    optimizer.zero_grad(set_to_none=True)

            # ---------------- VALIDATION ----------------

            model.eval()

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

                        loss = criterion(output, idxs)

                        total_valid_loss += loss.item()

            # Average loss per batch
            train_loss = (
                total_train_loss /
                len(train_dataloader)
            )

            valid_loss = (
                total_valid_loss /
                len(valid_dataloader)
            )

            print(
                f"Train loss: {train_loss:.4f} "
                f"Valid loss: {valid_loss:.4f}"
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:

        try:

            logging.info(
                "Entered initiate_model_trainer method"
            )

            # ---------------- LOAD DATASETS ----------------

            train_dataset = load_object(
                self.data_transformation_artifacts
                .train_transformed_object
            )

            valid_dataset = load_object(
                self.data_transformation_artifacts
                .valid_transformed_object
            )

            logging.info("Loaded datasets")

            # ---------------- DATALOADERS ----------------

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

            logging.info("Created dataloaders")

            # ---------------- MODEL ----------------

            model = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT
            )

            logging.info(
                "Loaded pretrained ResNet34"
            )

            # Replace final classification layer

            model.fc = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(
                    model.fc.in_features,
                    self.data_transformation_artifacts.classes
                )
            )

            logging.info(
                "Updated final layer"
            )

            model = model.to(DEVICE)

            # ---------------- LOSS ----------------

            criterion = nn.CrossEntropyLoss()

            # ---------------- OPTIMIZER ----------------

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=0.9
            )

            logging.info(
                "Model training started"
            )

            # ---------------- TRAINING ----------------

            for epoch in range(self.epochs):

                print(
                    f"Epoch {epoch + 1}/{self.epochs}"
                )

                logging.info(
                    f"Training epoch {epoch + 1}"
                )

                self.train(
                    model,
                    criterion,
                    optimizer,
                    train_loader,
                    valid_loader
                )

            logging.info(
                "Model training completed"
            )

            # ---------------- SAVE MODEL ----------------

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
                f"Saved trained model at "
                f"{self.model_trainer_config.TRAINED_MODEL_PATH}"
            )

            # ---------------- CREATE ARTIFACT ----------------

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=
                self.model_trainer_config.TRAINED_MODEL_PATH
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