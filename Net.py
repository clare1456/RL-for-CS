from utils.baseImport import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_efgat_v1 import *
from models.GAT import *

import copy

class Actor(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_dim: int = 128,
        device: Union[str, int, torch.device] = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = copy.deepcopy(preprocess_net)
        self.input_dim = self.preprocess.embed_dim
        self.output_dim = 2
        self.hidden_dim = hidden_dim
        self.last = MLP(
            self.input_dim, 
            self.output_dim, 
            self.hidden_dim, 
            device=self.device
        )

    def forward(
        self,
        state: Union[np.ndarray, torch.Tensor],
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        embeddings = F.relu(self.preprocess(state))
        logits = self.last(embeddings)
        probs = F.softmax(logits, dim=-1) 
        return probs


class Critic(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_dim: int = 128, 
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = copy.deepcopy(preprocess_net)
        self.input_dim = self.preprocess.embed_dim
        self.output_dim = 2
        self.hidden_dim = hidden_dim
        self.last = MLP(
            self.input_dim,  
            self.output_dim,
            self.hidden_dim,
            device=self.device
        )

    def forward(
        self, state: Union[np.ndarray, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor:
        """Mapping: s -> V(s)."""
        embeddings = F.relu(self.preprocess(state))
        return self.last(embeddings)
    

class Actor_choose_one(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_dim: int = 128,
        device: Union[str, int, torch.device] = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.input_dim = self.preprocess.embed_dim
        self.output_dim = 1
        self.hidden_dim = hidden_dim
        self.last = MLP(
            self.input_dim, 
            self.output_dim, 
            self.hidden_dim, 
            device=self.device
        )

    def forward(
        self,
        state: Union[np.ndarray, torch.Tensor],
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        embeddings = F.relu(self.preprocess(state))
        logits = self.last(embeddings).squeeze(-1)
        probs = F.softmax(logits, dim=-1) 
        return probs


class Critic_choose_one(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_dim: int = 128, 
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.input_dim = self.preprocess.embed_dim
        self.output_dim = 1
        self.hidden_dim = hidden_dim
        self.last = MLP(
            self.input_dim,  
            self.output_dim,
            self.hidden_dim,
            device=self.device
        )
    
    def forward(
        self, state: Union[np.ndarray, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor:
        """Mapping: s -> V(s)."""
        embeddings = F.relu(self.preprocess(state))
        return self.last(embeddings).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic 
        

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, device="cpu"):
        super().__init__()
        self.device = device
        self.process = nn.Sequential(
            nn.Linear(input_dim, hidden_dim).to(device), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, output_dim).to(device), 
        )
        self.process.to(device=device)
    
    def forward(self, x):
        if isinstance(x, torch.Tensor) == False:
            x = torch.Tensor(x).to(self.device)
        return self.process(x)
