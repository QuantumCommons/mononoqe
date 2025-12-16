import torch

from mononoqe.data.training_data import TrainingData
from mononoqe.training import Trainer
from mononoqe.utils.factory import get_factory


gpu_device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {gpu_device}")

def main():
    # Dataset
    training_data = TrainingData(batch_size=32, device=gpu_device)
    train_loader, val_loader, input_shape, output_shape = training_data.build_loaders()
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")
    print(f"Train loader length: {len(train_loader)}")
    print(f"Validation loader length: {len(val_loader)}")

    # Model
    factory = get_factory("artorius")
    print(f"Factory: {factory}")

    # Hyperparameters

    # Training
    trainer = Trainer()
    print(f"Trainer: {trainer}")
    # trainer.fit(
    #     model=model,
    #     topology_params=topology_params,
    #     training_params=training_params,
    #     training_data=training_data,
    #     device=gpu_device,
    # )


if __name__ == "__main__":
    main()
