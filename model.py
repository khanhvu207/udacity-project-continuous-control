import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
	def __init__(self, state_size, action_size, seed):
		super(ActorNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)

		self.fc1 = nn.Linear(state_size, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, action_size)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.tanh(self.fc4(x))
		return x

class CriticNetwork(nn.Module):
	def __init__(self, state_size, action_size, seed):
		super(CriticNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)

		self.fc1 = nn.Linear(state_size + action_size, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, 1)

	def forward(self, state, action):
		x = torch.cat((state, action), dim=1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x