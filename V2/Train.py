import wandb
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from U_net import SimpleUNet
from Dataset import PuzzleImageToImageDataset

# Initialize Weights & Biases
wandb.init(project="TheWitnessSolver", name="UNet_Training")

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PuzzleImageToImageDataset("C:\\Users\\jaopa\\Documents\\TheWitnessSolver\\Unsolved", "C:\\Users\\jaopa\\Documents\\TheWitnessSolver\\Solved", augment=False)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = SimpleUNet().to(device)  # Move the model to the GPU
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Log model configuration to wandb
wandb.watch(model, log="all")

for epoch in range(200):
    epoch_loss = 0
    for input_img, target_img in loader:
        # Move input and target images to the GPU
        input_img, target_img = input_img.to(device), target_img.to(device)

        pred = model(input_img)

        loss_fn = torch.nn.L1Loss()
        loss = loss_fn(pred, target_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

    # Log the loss to wandb
    wandb.log({"epoch": epoch, "loss": avg_loss})

# Save the model state after training
torch.save(model.state_dict(), f"TheWitnessSolverAI.pth")

# Finish the wandb run
wandb.finish()
