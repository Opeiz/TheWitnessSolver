import os    
import torch

from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from U_net import SimpleUNet  

# test_folder = "C:\\Users\\jaopa\\Documents\\TheWitnessSolver\\Test"
test_folder = "C:\\Users\\jaopa\\Documents\\TheWitnessSolver\\Unsolved"
output_folder = "C:\\Users\\jaopa\\Documents\\TheWitnessSolver\\Predictions"  
os.makedirs(output_folder, exist_ok=True) 


model = SimpleUNet()  
model_path = "C:\\Users\\jaopa\\Documents\\TheWitnessSolver\\TheWitnessSolverAI.pth"  
model.load_state_dict(torch.load(model_path))
model.eval()

# Perform predictions
with torch.no_grad():

    # Example dataset definition (replace with your actual dataset)
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = [transform(Image.open(os.path.join(test_folder, img_path)).convert("RGB")) for img_path in os.listdir(test_folder) if img_path.endswith(('.png', '.jpg', '.jpeg'))]

    for idx, input_img in enumerate(dataset):
        input_batch = input_img.unsqueeze(0)
        output = model(input_batch)
        save_image(output, os.path.join(output_folder, f"predicted_solution_{idx}.png"))
