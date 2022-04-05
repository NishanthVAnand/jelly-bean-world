from __future__ import absolute_import, division, print_function

try:
	import gym
	from gym import spaces, logger
	modules_loaded = True
except:
	modules_loaded = False
import gc

import numpy as np
from skimage.util.shape import view_as_windows

from .agent import Agent
from .direction import RelativeDirection
from .simulator import Simulator
from .visualizer import MapVisualizer


if not modules_loaded:
	__all__ = []
else:
	__all__ = ['JBWEnvCOMP579v1']

	class JBWEnvCOMP579v1(gym.Env):
		"""JBW environment for OpenAI gym.

		The action space consists of three actions:
		- `0`: Move forward.
		- `1`: Turn left.
		- `2`: Turn right.

		The observation space consists:
		- `scent`: Vector with shape 3
		- `object features`: Vector of size 512
		- `moved`: Binary value indicating whether the last 
			action resulted in the agent moving.

		After following the instructions provided in the main 
		`README` file to install the `jbw` framework, and 
		installing `gym` using `pip install gym`, this 
		environment can be used as follows:

		```
		import gym
		import jbw

		# Use 'JBW-render-COMP579' to include rendering support.
		# Otherwise, use 'JBW-COMP579', which should be much faster.
		env = gym.make('JBW-render-COMP579', render=True)

		# The created environment can then be used as any other 
		# OpenAI gym environment. For example:
		for t in range(10000):
		# Render the current environment.
		env.render(mode="matplotlib")
		# Sample a random action.
		action = env.action_space.sample()
		# Run a simulation step using the sampled action.
		scent, vision, moved, reward, _, _ = env.step(action)
		```
		"""

		def __init__(
			self, sim_config, reward_fn, ftype="hash", render=False):
			"""Creates a new JBW environment for OpenAI gym.

			Arguments:
				sim_config(SimulatorConfig) Simulator configuration
											to use.
				reward_fn(callable)         Function that takes the 
											previously collected 
											items and the current 
											collected items as inputs
											and returns a reward 
											value.
				render(bool)                Boolean value indicating 
											whether or not to support 
											rendering the 
											environment.
			"""
			self.sim_config = sim_config
			self._sim = None
			self._painter = None
			self._reward_fn = reward_fn
			self._render = render
			
			self.hash_dict = {(0,0,0):0, (1,0,0):1, (1,1,0):2, (0,0,1):3, (0,1,1):4}#, (0,1,0):5}
			self.shape_set = [(2,2), (3,3), (4,4), (5,5)]
			self.epi_len = 5000
			self.get_features = self.feature_picker(ftype)

			self.reset()

			# Computing shapes for the observation space.
			scent_shape = [len(self.sim_config.items[0].scent)]
			vision_dim = len(self.sim_config.items[0].color)
			vision_range = self.sim_config.vision_range
			vision_shape = [
				2 * vision_range + 1, 
				2 * vision_range + 1, 
				vision_dim]

			min_float = np.finfo(np.float32).min
			max_float = np.finfo(np.float32).max
			min_scent = min_float * np.ones(scent_shape)
			max_scent = max_float * np.ones(scent_shape)
			min_vision = min_float * np.ones(vision_shape)
			max_vision = max_float * np.ones(vision_shape)

			# Observations in this environment consist of a scent 
			# vector, a vision matrix, and a binary value 
			# indicating whether the last action resulted in the 
			# agent moving.
			self.scent_space = spaces.Box(low=min_scent, high=max_scent)
			self.vision_space = spaces.Box(low=min_vision, high=max_vision)
			#self.feature_space = spaces.Box(low=np.zeros(self.t_size), high=np.ones(self.t_size)) 
			self.feature_space = spaces.Box(low=np.zeros(self.t_size), high=np.ones(self.t_size))
			self.moved_space = spaces.Discrete(2)

			# There are three possible actions:
			#   1. Move forward,
			#   2. Turn left,
			#   3. Turn right.
			self.action_space = spaces.Discrete(4)

		def convert(self, vector):
			vector = np.ceil(vector)
			tuple_vec = tuple(vector)
			channel = self.hash_dict[tuple_vec]
			return channel

		def feature_picker(self, type="hash"):
			if type == "hash":
				self.t_size = 2048
				def feature_func(vision_state):
					features = []
					obs_channel = np.apply_along_axis(self.convert, 2, vision_state)
					obs_hash = obs_channel.choose(self.hash_vals)
					for s in self.shape_set:
						ids = np.bitwise_xor.reduce(view_as_windows(obs_hash, s), axis=(2,3)).flatten()%self.t_size
						features.extend(list(ids))
					feature_state = np.zeros(self.t_size)
					feature_state[list(set(features))] = 1
					return feature_state
				
			elif type == "obj":
				self.t_size = 900
				def feature_func(vision_state):
					features = []
					obs_channel = np.apply_along_axis(self.convert, 2, vision_state)
					obs_channel = obs_channel.flatten()
					features = np.zeros((obs_channel.size, len(self.hash_dict)))
					features[np.arange(obs_channel.size), obs_channel] = 1
					return features[:, 1:].flatten()

			else:
				raise NotImplementedError

			return feature_func

		def step(self, action):
			"""Runs a simulation step.

			Arguments:
				action(int) Action to take, which can be one of:
							- `0`: Move forward.
							- `1`: Turn left.
							- `2`: Turn right.

			Returns:
				observation (dictionary): Contains:
					- `scent`: Vector with shape `[S]`, where `S` 
					is the scent dimensionality.
					- `vision`: Matrix with shape 
					`[2R+1, 2R+1, V]`, where `R` is the vision 
					range and `V` is the vision/color 
					dimensionality.
					- `moved`: Binary value indicating whether the 
					last action resulted in the agent moving.
				reward (float): Amount of reward obtained from the 
					last action.
				done (bool): Whether or not the episode has ended 
					which is always `False` for this environment.
				info (dict): Empty dictionary.
			"""

			if self.T == self.epi_len:
				raise ValueError("Reset environment before calling the step function")

			prev_position = self._agent.position()
			prev_items = self._agent.collected_items()

			self._agent._next_action = action
			self._agent.do_next_action()

			position = self._agent.position()
			items = self._agent.collected_items()
			reward = self._reward_fn(prev_items, items)
			done = self.T+1 >= self.epi_len

			self.scent_state = self._agent.scent()
			self.vision_state = self._agent.vision()
			self.feature_state = self.get_features(self.vision_state)
			self.moved_state = np.any(prev_position != position)

			self.T += 1

			return (self.scent_state, self.vision_state, self.feature_state, self.moved_state), reward, done, {}

		def reset(self):
			"""Resets this environment to its initial state."""
			self.T = 0
			del self._sim
			gc.collect()
			self._sim = Simulator(sim_config=self.sim_config)
			self._agent = _JBWEnvAgent(self._sim)
			self.hash_vals = np.random.randint(0, np.iinfo(np.int32).max,\
				size=(1+len(self.sim_config.items),(2*self.sim_config.vision_range)+1,(2*self.sim_config.vision_range)+1))
			self.hash_vals[0,:,:] = 0
			if self._render:
				del self._painter
				self._painter = MapVisualizer(
				self._sim, self.sim_config, 
				bottom_left=(-70, -70), top_right=(70, 70))
			self.scent_state = self._agent.scent()
			self.vision_state = self._agent.vision()
			self.feature_state = self.get_features(self.vision_state)
			self.moved_state = False
			return (self.scent_state, self.vision_state, self.feature_state, self.moved_state)

		def render(self, mode='matplotlib'):
			"""Renders this environment in its current state.
			 Note that, in order to support rendering, 
			`render=True` must be passed to the environment 
			constructor.

			Arguments:
				mode(str) Rendering mode. Currently, only 
						`"matplotlib"` is supported.
			"""
			if mode == 'matplotlib' and self._render:
				self._painter.draw()
			elif not self._render:
				logger.warn(
				'Need to pass `render=True` to support '
				'rendering.')
			else:
				logger.warn(
				'Invalid rendering mode "%s". '
				'Only "matplotlib" is supported.')

		def close(self):
			"""Deletes the underlying simulator and deallocates 
			all associated memory. This environment cannot be used
			again after it's been closed."""
			del self._sim
			return

		def seed(self, seed=None):
			self.sim_config.seed = seed
			np.random.seed(seed)
			self.reset()
			return


class _JBWEnvAgent(Agent):
	"""Helper class for the JBW environment, that represents
	a JBW agent living in the simulator.
	"""

	def __init__(self, simulator):
		"""Creates a new JBW environment agent.
		
		Arguments:
			simulator(Simulator)  The simulator the agent lives in.
		"""
		super(_JBWEnvAgent, self).__init__(
			simulator, load_filepath=None)
		self._next_action = None

	def do_next_action(self):
		if self._next_action == 0:
			self.move(RelativeDirection.FORWARD)
		elif self._next_action == 1:
			self.move(RelativeDirection.LEFT)
		elif self._next_action == 2:
			self.move(RelativeDirection.RIGHT)
		elif self._next_action == 3:
			self.move(RelativeDirection.BACKWARD)
		else:
			logger.warn(
			'Ignoring invalid action %d.' 
			% self._next_action)

	# There is no need for saving and loading an agent's
	# state, as that can be done outside the gym environment.

	def save(self, filepath):
		pass

	def _load(self, filepath):
		pass
