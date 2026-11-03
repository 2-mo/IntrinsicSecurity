"""Check all four GPU ranks can exchange data before starting the model server."""
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
assert value.item() == 10.0, (rank, value.item())
print(f'NCCL CHECK PASSED rank={rank} sum={value.item()}', flush=True)
dist.destroy_process_group()
