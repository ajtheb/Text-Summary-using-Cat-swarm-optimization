import random
import numpy as np
from enum import Enum

import simulation

class Behavior(Enum):
	SEEKING = 1
	TRACING = 2


class Cat:
	def __init__(self, behavior, position, velocities, vmax):
		self.behavior = behavior
		self._position = position
		self._velocities = velocities
		self._vmax = vmax
		self._dimension_size = len(self._position)

	def evaluate(self, function):
		return function(self._position), self._position

	def move(self, function, best_pos):
		if self.behavior == Behavior.SEEKING:
		#------SEEKING------
			candidate_moves = []

			for j in range(simulation.SMP):
				candidate_moves.append(
					[
						random.uniform(
							self._position[idx_dim] - (self._position[idx_dim] * simulation.SRD) / 100, 
							self._position[idx_dim] + (self._position[idx_dim] * simulation.SRD) / 100
						)
						for idx_dim in range(self._dimension_size)
					]
				)
			
			fitness_values = [function(candidate) for candidate in candidate_moves]

			fit_min = min(fitness_values)
			fit_max = max(fitness_values)

			probabilities = [abs(value - fit_max) / (fit_max - fit_min) for value in fitness_values]
			prob_sum = sum(probabilities)
			probabilities = list(map(lambda prob: (float)(prob / prob_sum), probabilities))

			next_position_idx = np.random.choice(simulation.SMP, 1, p=probabilities)[0]
			self._position = candidate_moves[next_position_idx]
		elif self.behavior == Behavior.TRACING:
			#------TRACING------
			r1 = random.random()

			for idx_dim in range(self._dimension_size):
				#Compute velocity
				self._velocities[idx_dim] = self._velocities[idx_dim] + r1 * simulation.c1 * (best_pos[idx_dim] - self._position[idx_dim])
				#Apply bounds			
				self._velocities[idx_dim] = min(self._velocities[idx_dim], self._vmax)
				self._velocities[idx_dim] = max(self._velocities[idx_dim], -self._vmax)
				#Move with computed velocity
				self._position[idx_dim] = self._position[idx_dim] + self._velocities[idx_dim]

			
		else:
			raise Exception("Unreachable")