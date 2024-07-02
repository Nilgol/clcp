# Example usage
root_path = '/homes/math/golombiewski/workspace/data/A2D2'
config_path = '/homes/math/golombiewski/workspace/data/A2D2_general/cams_lidars.json'
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add any other image transformations here
])
dataset = A2D2Dataset(root_path, config_path, transform=transform)

# Creating a DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Example of using the dataloader
for batch in dataloader:
    images, point_clouds = batch
    # Process the batch
    print(images.shape, point_clouds.shape)