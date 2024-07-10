import torch
from torch.utils.data import DataLoader


def collate_fn(batch):
    images = [item[0] for item in batch]
    point_clouds = [item[1] for item in batch]

    images = torch.stack(images)
    return images, point_clouds


def build_loader(dataset, batch_size=32, num_workers=16, shuffle=True):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return dataloader


if __name__ == "__main__":
    batch_size = 8
    num_workers = 4

    # Initialize the DataLoader
    dataloader = build_loader(
        batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    print(
        f"DataLoader initialized with batch size {batch_size} and {num_workers} workers."
    )

    # Fetch a few batches and print their shapes
    for i, (images, point_clouds) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Number of point clouds: {len(point_clouds)}")
        print(
            f"  First point cloud shape: {point_clouds[0].shape if len(point_clouds) > 0 else 'N/A'}"
        )
        if i == 2:  # Stop after 3 batches
            break
