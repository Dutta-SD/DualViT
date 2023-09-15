# import torch

# ckpt_full_path="./checkpoints/vit-b-16-modified-loss-updated-google-weights_1694713704.pt"

# ckpt = torch.load(ckpt_full_path)

# torch.save(
#     {
#         "model": ckpt["model"],
#         "opt": ckpt["opt"],
#         "tags": ckpt["tags"],
#         "best_train_score": 0.3763749897480011,
#         "best_test_score": 0.3055555522441864,
#     },
#     ckpt_full_path,
# )
