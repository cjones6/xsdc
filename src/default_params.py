import numpy as np
import random
import torch

# Even with setting the seeds below there can still be non-deterministic behavior.
# See https://pytorch.org/docs/stable/notes/randomness.html

device = torch.device('cuda:0')
# device = torch.device('cpu')
torch.set_default_tensor_type(torch.DoubleTensor)

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

dataloader_timeout = 0