## RL-Final_Project
### Farmer's helper! :man_farmer: :helicopter: 

#### *The goal* of our project is a drone with a watering system uses Q-learning  to irrigate the farmland until the soil moisture reaches sufficient moisture. In order to solve the problem of food crisis in the future and can reduce the manpower.<br>

#### Clone the package **Final_Project** in our github,  `cd gym-examples` and  `pip install - e .` 
#### Our project use Open AI gym toolkit to built environment and  pygame to display. Please `pip install gym` and `pip install pygam`<br>
***

#### *Environment : Farm which has the moisture sensor.*
#### *Agent : The Drone (blue circle) with watering system.*
#### *Actions : right, up ,left ,down ,watering.*
#### *State : The state of agent * The state of ground moisture * The state of tank capacity.*
#### The environment I built is a four by four farmland with a pond in it. The degree of dark color represents the soil humidity.<br/>*(The location of the drone and the humidity of the soil will be reset randomly in each episode.)*

<img src="env.jpg" alt="" width="300" heigh="300"/>
 
 ***
 ###Reward
 





***
### Problem 
- **Memory Error** — The lager state let to a memory error that can not be fixed. If the state of the farm size more than four by four or the state of ground moisture more than two. The total state will be too large to estimate.I think changing the algorithm to DQN can improve this problem.
- **Observation Space Not Enough** — The agent will miss the goal if its observation space does not contain the tank.  I think it might be caused by the agent not knowing when to refill the tank. I solved this problem after considering the state of the tank for the agent.
