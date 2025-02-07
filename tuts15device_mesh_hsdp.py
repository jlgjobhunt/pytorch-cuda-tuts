# How to use DeviceMesh with HSDP.
# URL_for_Citation: https://pytorch.org/tutorials/recipes/distributed_device_mesh.html#how-to-use-devicemesh-with-hsdp
# 
# Hybrid Sharding Data Parallel (HSDP) is a 2D strategy to perform
# FSDP within a host and DDP across hosts.
#
# Let's see an example of how DeviceMesh can assist with applying
# HSDP to your model with a simple setup.
# With DeviceMesh, users would not need to manually create and
# manage shard group and replicate group.
#
#
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
    

# HSDP: MeshShape(2, 4)
mesh_2d = init_device_mesh("cuda", (2, 4))
model = FSDP(
    ToyModel().to("cuda:0"), device_mesh=mesh_2d, sharding_strategy=ShardingStrategy.HYBRID_SHARD
)

# Add this line to shut down the process group.
torch.distributed.destroy_process_group()


