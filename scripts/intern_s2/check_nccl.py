"""Check all configured GPU ranks can exchange data before loading weights."""
from datetime import timedelta
import os

import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=timedelta(seconds=60))
value = torch.tensor([rank + 1.0], device=f'cuda:{rank}')
dist.all_reduce(value)
torch.cuda.synchronize()
world_size = dist.get_world_size()
expected = world_size * (world_size + 1) / 2
assert value.item() == expected, (rank, value.item(), expected)
print(f'NCCL CHECK PASSED rank={rank} sum={value.item()}', flush=True)
dist.destroy_process_group()
