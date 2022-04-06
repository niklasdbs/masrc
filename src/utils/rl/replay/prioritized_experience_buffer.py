import numpy as np

from utils.rl.replay.transition import Transition


class SumTree:
    def __init__(self, capacity : int):
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity - 1)

    def _propagate(self, index, delta):
        parent_idx = (index - 1) // 2

        self._tree[parent_idx] += delta

        if parent_idx != 0:
            self._propagate(parent_idx, delta)


    def update(self, index, p):
        delta = p - self._tree[index]
        self._tree[index] = p
        self._propagate(index, delta)

    def total(self):
        return self._tree[0]

    def _retrieve(self, index, s):
        left = 2 * index + 1
        right = left + 1

        if left >= len(self._tree):
            return index

        # if self._tree[left] == self._tree[right]:
        #     return self._retrieve(np.random.choice([left, right]), s)

        if s <= self._tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self._tree[left])

    def get(self, s):
        index = self._retrieve(index=0, s=s)
        data_index = index - self.capacity + 1

        return index, self._tree[index], data_index

class PrioritizedExperienceBuffer:
    def __init__(self, config):
        self.max_capacity = config.replay_size
        self._full = False
        self._data = np.empty(self.max_capacity, dtype=Transition)
        self._next_index = 0
        self._sum_tree = SumTree(self.max_capacity)
        self._max_priority = 1.0
        self._alpha = 0.6
        self._epsilon = 1e-6
        self._beta = 0.4
        self._beta_increase = 0.00001

    def add_transition(self, transition : Transition, *args, **kwargs):
        self._data[self._next_index] = transition
        self._sum_tree.update(self._next_index + self._sum_tree.capacity - 1, self._max_priority)
        self._next_index = self._next_index + 1
        if self._next_index == self.max_capacity:
            self._next_index = 0
            self._full = True


    def update(self, indices, errors):
        prios = self.get_priority(errors)
        for idx, priority in zip(indices, prios):
            self._sum_tree.update(idx, priority)

        self._max_priority = max(self._max_priority, np.max(prios))

    def get_priority(self, error):
        return (error + self._epsilon) ** self._alpha


    def size(self):
        return self.max_capacity if self._full else self._next_index

    def sample(self, n_samples: int):
        states = [None] * n_samples
        actions = [None] * n_samples
        rewards = [None] * n_samples
        dones = [None] * n_samples
        infos = [None] * n_samples
        next_states = [None] * n_samples

        total_p = self._sum_tree.total()
        segment  = total_p / n_samples

        priorities = np.zeros(n_samples, dtype=np.float32)
        indices = np.zeros(n_samples, dtype=np.int)
        a = np.arange(n_samples) * segment
        b = np.arange(1, n_samples + 1) * segment
        samples = np.random.uniform(a, b)
        for i, s in enumerate(samples):
            index, prio, data_index = self._sum_tree.get(s)
            transition = self._data[data_index]

            states[i] = transition.state
            actions[i] = transition.action
            rewards[i] = np.float32(transition.reward)
            dones[i] = transition.done
            infos[i] = transition.info["dt"]  # todo handle case without dt
            next_states[i] = transition.next_state

            priorities[i] = prio
            indices[i] = index

        sampling_probabilities = priorities / total_p
        is_weights = (self.size() * sampling_probabilities) ** -self._beta #todo beta schedule
        is_weights = is_weights/is_weights.max()

        batch = {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "infos": infos,
                "next_states": next_states,
                "indices": indices,
                "is_weights": is_weights
        }

        self._beta = min(1.0, self._beta + self._beta_increase)

        return batch