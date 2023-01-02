import gym
from gym import spaces
import pygame
import numpy as np
import os

class Fly(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=4):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, 3, shape=(2,), dtype=int),
                "wet": spaces.Box(0, 1, shape=(15,), dtype=int),
                "water":spaces.Box(0, 9, shape=(1,), dtype=int),
            }
        )

        # We have 5 actions, corresponding to "right", "up", "left", "down", "watering"
        self.action_space = spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),    #right
            1: np.array([0, 1]),    #up
            2: np.array([-1, 0]),   #left
            3: np.array([0, -1]),   #down
            4: np.array([0, 0]),    #watering
        }
        
        self.water = 9  #the capacity of tank

        self.space = np.arange(0, 16 ,1).reshape([4,4])

        self.perfact = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        self._water_location = np.array([0, 0])
        
        self.nplinspace = np.arange(0, 524288 * 10, 1).reshape([4,4,10,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
    
        self.observation_space_state = 524288* 10
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "water": self.water, "wet": self._wet_Number}

    def _get_info(self):
        return {
            "wet": np.linalg.norm(
                self._agent_location - self._water_location, ord=1
            )
        }
    
    def s2s_state(self, s):
        return self.nplinspace[s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8],s[9],s[10],s[11],s[12],s[13],s[14],s[15],s[16],s[17]]

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.water = 9

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, 4, size=2, dtype=int)
        
        # Generate random humidity of ground
        self._wet_Number = self.np_random.integers(0, 2, size=15, dtype=int)

        while (np.sum(self._wet_Number) > 4):
            self._wet_Number = self.np_random.integers(0, 2, size=15, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Step Function
        reward = 0

        # Map the action (element of {0,1,2,3,4}) to the direction we walk in
        direction = self._action_to_direction[action]
        
        before_agent_location = self._agent_location.copy()
        # We use `np.clip` to make sure we don't leave the grid 
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if np.array_equal(before_agent_location, self._agent_location) and action != 4:
            reward = -100

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._wet_Number, self.perfact)

        if terminated:
            reward = 1000
            
        #describe the agent location 
        agent_location = self.space[self._agent_location[0],self._agent_location[1]]
        
        # choses watering and agent not at the pound
        if agent_location > 0 and action == 4:
            agent_location = agent_location - 1         
            # if water in tank
            if self.water > 0:  
                
                # if the ground not moisture enough
                if self._wet_Number[agent_location] <= 0:
                    
                    # can watering    
                    self._wet_Number[agent_location] += 1
                    self.water = self.water - 1
                    reward = 50
                # if moisture enough
                else:
                    # can not watering
                    reward = -100
            else:
                # if no water in tank
                reward = -100

        #refill tank
        if np.array_equal(self._agent_location, self._water_location):   
  
            if self.water <= 2:
                reward = 10
            else:
                reward = -20
            self.water = 9

        reward = reward + (-1 + np.sum(self._wet_Number) / 15) #encourages the agent to watering

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

# draw the window
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        self.pix_square_size = self.window_size / self.size

        pygame.draw.rect(
            canvas,
            (154, 205, 50),
            pygame.Rect(
                pix_square_size * self._water_location,
                (pix_square_size, pix_square_size),
            ),
        )

        #drow pond
        self.position_xp=self._water_location[0]
        self.position_yp=self._water_location[1]
        pond_image=pygame.image.load(os.path.abspath("pond_with_fish.png"))
        pond_image=pygame.transform.scale(pond_image,(self.pix_square_size,self.pix_square_size))
        
        
        canvas.blit(pond_image,((self.pix_square_size)*self.position_xp,(self.pix_square_size)*self.position_yp))
        
       

        pygame.display.update()
        


        # draw the moisture of ground
        for i in np.argwhere(self.space > 0):
            # print(self.space[i[0], i[1]] - 1)
            # print(self._wet_Number[self.space[i[0], i[1]] - 1])     # 濕度
            if self._wet_Number[self.space[i[0], i[1]] - 1] == 0:
                pygame.draw.rect(
                    canvas,
                    (187, 255, 255),
                    pygame.Rect(
                        pix_square_size * i,
                        (pix_square_size, pix_square_size),
                    ),
                )
            elif self._wet_Number[self.space[i[0], i[1]] - 1] == 1:
                pygame.draw.rect(
                    canvas,
                    (0, 229, 238),
                    pygame.Rect(
                        pix_square_size * i,
                        (pix_square_size, pix_square_size),
                    ),
                )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
