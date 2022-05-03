from __future__ import absolute_import, division, print_function

import numpy as np
try:
	from gym.envs.registration import register
	modules_loaded = True
except:
	modules_loaded = False

from .agent import Agent
from .direction import RelativeDirection
from .item import *
from .simulator import *
from .visualizer import MapVisualizer, pi

def make_config():
	items = []
	items.append(Item("apple", [1.64, 0.54, 0.4], [1.0, 0.0, 0.0], [0, 0,  0, 0], [0, 0,  0, 0], False, 0.0,
			intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-0.5],
			interaction_fns=[[InteractionFunction.PIECEWISE_BOX, 10.0, 100.0, 0.0, -6.0],\
							[InteractionFunction.ZERO],\
							#[InteractionFunction.ZERO],\
							[InteractionFunction.PIECEWISE_BOX, 10.0, 100.0, -100.0, -100.0],\
							[InteractionFunction.PIECEWISE_BOX, 10.0, 100.0, 0.0, -6.0]]))

	items.append(Item("banana", [1.92, 1.76, 0.4], [1.0, 1.0, 0.0], [0, 0, 0, 0], [0, 0, 0, 0], False, 0.0,
			intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-5.5],
			interaction_fns=[[InteractionFunction.ZERO],\
							[InteractionFunction.ZERO],\
							#[InteractionFunction.ZERO],\
							[InteractionFunction.ZERO],\
							[InteractionFunction.ZERO]]))

	# items.append(Item("walls", [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], True, 0.0,
	# 		intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-10],
	# 		interaction_fns=[[InteractionFunction.ZERO],\
	# 						[InteractionFunction.ZERO],\
	# 						[InteractionFunction.CROSS, 20,40,8,-1000,-1000,-1],\
	# 						[InteractionFunction.ZERO],\
	# 						[InteractionFunction.ZERO]]))

	items.append(Item("jellybean", [0.68, 0.01, 0.99], [0.0, 0.0, 1.0], [0, 0,  0, 0], [0, 0,  0, 0], False, 0.0,
			intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-0.5],
			interaction_fns=[[InteractionFunction.PIECEWISE_BOX, 10.0, 100.0, -100.0, -100.0],\
							[InteractionFunction.ZERO],\
							#[InteractionFunction.ZERO],\
							[InteractionFunction.PIECEWISE_BOX, 10.0, 100.0, 0.0, -6.0],\
							[InteractionFunction.PIECEWISE_BOX, 10.0, 100.0, -100.0, -100.0]]))

	items.append(Item("truffle", [8.4, 4.8, 2.6], [0.0, 1.0, 1.0], [0, 0, 0,  0], [0, 0,  0, 0], False, 0.0,
			intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-2.0],
			interaction_fns=[[InteractionFunction.PIECEWISE_BOX, 10.0, 100.0, 0.0, -6.0],\
							[InteractionFunction.ZERO],\
							#[InteractionFunction.ZERO],\
							[InteractionFunction.PIECEWISE_BOX, 10.0, 100.0, -100.0, -100.0],\
							[InteractionFunction.PIECEWISE_BOX, 10.0, 100.0, 0.0, -6.0]]))

	# construct the simulator configuration
	config = SimulatorConfig(max_steps_per_movement=1, vision_range=7,
	  allowed_movement_directions=[ActionPolicy.ALLOWED, ActionPolicy.ALLOWED, ActionPolicy.ALLOWED, ActionPolicy.ALLOWED],
	  allowed_turn_directions=[ActionPolicy.DISALLOWED, ActionPolicy.DISALLOWED, ActionPolicy.DISALLOWED, ActionPolicy.DISALLOWED],
	  no_op_allowed=False, patch_size=32, mcmc_num_iter=4000, items=items, agent_color=[0.0, 0.0, 0.0],
	  collision_policy=MovementConflictPolicy.FIRST_COME_FIRST_SERVED, agent_field_of_view=2*pi,
	  decay_param=0.4, diffusion_param=0.14, deleted_item_lifetime=2000)

	return config

def make_reward():
	def get_reward(prev_items, items):
		reward_array = np.array([1, 0.1,  -1, -0.1])
		diff = items - prev_items
		return (diff * reward_array).sum().astype(np.float32)
	return get_reward

sim_config = make_config()
reward_fn = make_reward()

register(
	  id='JBW-COMP579-hash-Transfer-v1',
	  entry_point='jbw.env_COMP579_episodic_transfer:JBWEnvCOMP579Transferv1',
	  kwargs={
		'sim_config': sim_config,
		'reward_fn': reward_fn,
		'ftype' : "hash",
		'render': False})

register(
	  id='JBW-COMP579-render-hash-Transfer-v1',
	  entry_point='jbw.env_COMP579_episodic_transfer:JBWEnvCOMP579Transferv1',
	  kwargs={
		'sim_config': sim_config,
		'reward_fn': reward_fn,
		'ftype' : "hash",
		'render': True})


register(
	  id='JBW-COMP579-obj-Transfer-v1',
	  entry_point='jbw.env_COMP579_episodic_transfer:JBWEnvCOMP579Transferv1',
	  kwargs={
		'sim_config': sim_config,
		'reward_fn': reward_fn,
		'ftype' : "obj",
		'render': False})

register(
	  id='JBW-COMP579-render-obj-Transfer-v1',
	  entry_point='jbw.env_COMP579_episodic_transfer:JBWEnvCOMP579Transferv1',
	  kwargs={
		'sim_config': sim_config,
		'reward_fn': reward_fn,
		'ftype' : "obj",
		'render': True})