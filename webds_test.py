import webdataset as wds
import torch
from torchvision.transforms import ToTensor

normalize = normalize_transform()  # your function
def pil_to_tensor(pil):
    return normalize(ToTensor()(pil))

urls = "shards/train-{00000..00010}.tar"  # adjust range
dataset = (
    wds.WebDataset(urls, shardshuffle=True)
      .decode("pil")
      .to_tuple("png", "cls")
      .map_tuple(pil_to_tensor, lambda y: int(y))
)

loader = torch.utils.data.DataLoader(
    dataset.batched(batch_size, partial=False),
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)
