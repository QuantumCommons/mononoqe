import torch

from mononoqe.data.training_data import TrainingData
from mononoqe.models.topologies import TopologyParams, build_topology
from mononoqe.models.model import Net
from mononoqe.training.training_params import TrainingParams

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
    print("=" * 50)

    # Model
    topology_params = TopologyParams(name="mein", input_shape=input_shape, output_shape=output_shape)

    manual_topology = [
        {"type": "flatten", "start_dim": 0, "end_dim": -1},
        {"type": "linear", "output_size": 128, "bias": True},
        {"type": "relu", "inplace": True},
        {"type": "linear", "output_size": 10, "bias": True},
    ]

    build_topology(topology_params=topology_params, loaded_topology_list=manual_topology)

    model = Net()
    model.configure_topology(topology=topology_params)

    # Hyperparameters
    training_params = TrainingParams(learning_rate=1e-3, epochs=10, loss_name="cross_entropy", optimizer_name="adam")
    # model.configure_training(training_params=training_params)
    print(f"Model: {model}")
    print("=" * 50)

    # Training
    # trainer = Trainer()
    # print(f"Trainer: {trainer}")
    # trainer.fit(
    #     model=model,
    #     topology_params=topology_params,
    #     training_params=training_params,
    #     training_data=training_data,
    #     device=gpu_device,
    # )


if __name__ == "__main__":
    main()
