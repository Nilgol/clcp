import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(image_embeddings, lidar_embeddings, temperature):
    # Normalize the embeddings
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    lidar_embeddings = F.normalize(lidar_embeddings, p=2, dim=-1)

    # Compute cosine similarity matrix
    logits = (image_embeddings @ lidar_embeddings.T) / temperature

    # Labels are the indices of the correct pairs
    labels = torch.arange(image_embeddings.size(0), device=image_embeddings.device)

    # Cross entropy loss for both directions
    loss_i = F.cross_entropy(logits, labels)
    loss_l = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_l) / 2, logits, loss_i, loss_l


def get_embeddings(batch_size, embed_dim, mode="random"):
    if mode == "same_random":
        embeddings = torch.randn(batch_size, embed_dim)
        return embeddings, embeddings
    elif mode == "disturbed":
        base_embeddings = (
            torch.arange(batch_size * embed_dim).float().view(batch_size, embed_dim)
        )
        disturbed_embeddings = base_embeddings + torch.randn_like(base_embeddings) * 0.1
        return base_embeddings, disturbed_embeddings
    elif mode == "random":
        image_embeddings = torch.randn(batch_size, embed_dim)
        lidar_embeddings = torch.randn(batch_size, embed_dim)
        return image_embeddings, lidar_embeddings
    elif mode == "one_to_all":
        image_embeddings = torch.randn(batch_size, embed_dim)
        lidar_embeddings = torch.randn(batch_size, embed_dim)
        image_embeddings[0] = lidar_embeddings[0]  # Making one pair identical
        return image_embeddings, lidar_embeddings
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    batch_size = 4
    embed_dim = 8
    temperature = 0.07

    for mode in ["same_random", "disturbed", "random", "one_to_all"]:
        print(f"\nTesting mode: {mode}")
        image_embeddings, lidar_embeddings = get_embeddings(batch_size, embed_dim, mode)
        loss, logits, loss_i, loss_l = contrastive_loss(
            image_embeddings, lidar_embeddings, temperature
        )
        print(f"Loss: {loss.item()}")
        print(f"Logits:\n{logits}")
        print(f"Loss_i: {loss_i.item()}, Loss_l: {loss_l.item()}")
