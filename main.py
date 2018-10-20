from dqn_agent import Agent
from unityagents import UnityEnvironment
import numpy as np

#Mac:
#filename = "Banana.app"

#Windows (x86):
#file_name = Banana_Windows_x86/Banana.exe

#Windows (x86_64):
file_name = "Banana_Windows_x86_64/Banana.exe"

#Linux (x86):
#file_name = "Banana_Linux/Banana.x86"

#Linux (x86_64)
#file_name = "Banana_Linux/Banana.x86_64"

#Define the Unity environment
env = UnityEnvironment(file_name)

#Get the "brain" name of Banana Man
brain_name = env.brain_names[0]

#Create Agent
agent = Agent(environment = env, agent_name = brain_name, train_agent = True)

#Perform training. Will also save scores as '.csv' file
agent.train_dqn()

#Close the Unity environment. Friendly note: Whenever working with Unity's ML-agents, also ensure the environment is properly closed.
env.close()
