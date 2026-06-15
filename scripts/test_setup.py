import os
os.environ["MUJOCO_GL"] = "egl" 

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.factory import make_env
from lerobot.envs.configs import LiberoEnv

print("1. Testing Dataset Streaming...")
dataset = LeRobotDataset("lerobot/libero")
print(f"Dataset loaded! Number of episodes: {dataset.num_episodes}")

sample = dataset[0]
print(f"Observation keys: {sample.keys()}")
print(f"Image shape: {sample['observation.images.image'].shape}")

print("\n2. Testing Simulation Environment...")
cfg = LiberoEnv(task="libero_10")
# This returns a dictionary of 10 vector environments
envs_dict = make_env(cfg)

# Extract just Task 0 (the first task in the libero_10 suite)
suite_name = "libero_10"
task_id = 0
env = envs_dict[suite_name][task_id]

# Now we can reset the actual environment
obs, info = env.reset()
print(f"Environment reset successful! Task ID loaded: {task_id}")

# Because this is a VectorEnv (handles parallel environments), the action space 
# expects a batch dimension. We sample a single action and add a batch dimension [1, action_dim]
action = env.action_space.sample() 
obs, reward, done, truncated, info = env.step(action)

print("Dummy action executed successfully! Setup is complete.")

import torchvision.transforms as T

print(f"\nAvailable observation keys: {obs.keys()}")

# 1. Access the nested pixels dictionary
if "pixels" in obs:
    pixels_dict = obs["pixels"]
    print(f"Keys inside 'pixels': {pixels_dict.keys()}")
    
    # 2. Find the primary camera (usually 'agentview_image' or 'image')
    # Let's find any key inside 'pixels' containing image data
    img_key = next((k for k in pixels_dict.keys() if "image" in k or "pixels" in k), None)
    
    if img_key:
        img_tensor = pixels_dict[img_key]
        
        # LeRobot wrapper might keep the batch dim from VectorEnv [1, C, H, W]
        # or it might be [C, H, W]. Let's handle both safely:
        if img_tensor.ndim == 4:
            img_tensor = img_tensor[0]
            
        # Standardize axis order: if it's [H, W, C] (robosuite native), 
        # T.ToPILImage expects [C, H, W] or channels last depending on types. 
        # Let's print the shape to see how LeRobot wrapped it.
        print(f"Found image '{img_key}' with shape: {img_tensor.shape}")
        
        transform = T.ToPILImage()
        img = transform(img_tensor)
        
        output_filename = "robot_view_test.png"
        img.save(output_filename)
        print(f"Visual frame saved successfully to: {os.path.abspath(output_filename)}")
    else:
        print("Could not find a valid image camera key inside 'pixels'.")
else:
    print("Could not find 'pixels' key in the observation dictionary.")