"""
Custom environments for the RL notebooks.
Simple, transparent environments where you can see every moving part.
"""
import numpy as np


class GridEnvironment:
    """
    A simple NxM GridEnvironment environment.

    The agent starts at `start` and must reach `goal`.
    Some cells can be walls (impassable) or traps (negative reward).

    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT

    This is intentionally NOT using the Gymnasium API so you can see
    every piece of an environment from scratch. We'll wrap it later.
    """

    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ACTION_NAMES = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    def __init__(self, rows=4, cols=4, start=(0, 0), goal=(3, 3),
                 walls=None, traps=None, step_reward=-0.1, goal_reward=1.0,
                 trap_reward=-1.0, stochastic=False, slip_prob=0.1):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.walls = set(walls or [])
        self.traps = set(traps or [])
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.stochastic = stochastic
        self.slip_prob = slip_prob

        self.n_states = rows * cols
        self.n_actions = 4

        self.state = start
        self._build_transition_model()

    def _build_transition_model(self):
        """Pre-compute the full transition model P[s][a] = [(prob, s', reward, done)]."""
        self.P = {}
        for r in range(self.rows):
            for c in range(self.cols):
                s = self._to_index(r, c)
                self.P[s] = {}
                for a in range(self.n_actions):
                    self.P[s][a] = self._get_transitions(r, c, a)

    def _get_transitions(self, row, col, action):
        """Return list of (probability, next_state, reward, done) for a state-action pair."""
        if (row, col) == self.goal or (row, col) in self.traps:
            # Terminal state — stays in place
            s = self._to_index(row, col)
            return [(1.0, s, 0.0, True)]

        transitions = []
        if self.stochastic:
            # With prob (1-slip), go intended; with slip_prob, go random
            all_actions = list(range(self.n_actions))
            for a in all_actions:
                if a == action:
                    prob = 1.0 - self.slip_prob + self.slip_prob / self.n_actions
                else:
                    prob = self.slip_prob / self.n_actions
                nr, nc = self._next_pos(row, col, a)
                s_next = self._to_index(nr, nc)
                reward = self._get_reward(nr, nc)
                done = (nr, nc) == self.goal or (nr, nc) in self.traps
                transitions.append((prob, s_next, reward, done))
        else:
            nr, nc = self._next_pos(row, col, action)
            s_next = self._to_index(nr, nc)
            reward = self._get_reward(nr, nc)
            done = (nr, nc) == self.goal or (nr, nc) in self.traps
            transitions.append((1.0, s_next, reward, done))

        return transitions

    def _next_pos(self, row, col, action):
        dr, dc = self.ACTIONS[action]
        nr, nc = row + dr, col + dc
        # Clamp to grid boundaries and check for walls
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols or (nr, nc) in self.walls:
            return row, col  # Stay in place
        return nr, nc

    def _get_reward(self, row, col):
        if (row, col) == self.goal:
            return self.goal_reward
        if (row, col) in self.traps:
            return self.trap_reward
        return self.step_reward

    def _to_index(self, row, col):
        return row * self.cols + col

    def _to_rowcol(self, index):
        return divmod(index, self.cols)

    def reset(self):
        """Reset agent to start. Returns state index."""
        self.state = self.start
        return self._to_index(*self.state)

    def step(self, action):
        """Take an action. Returns (next_state, reward, done, info)."""
        transitions = self.P[self._to_index(*self.state)][action]

        # Sample from transition distribution
        probs = [t[0] for t in transitions]
        idx = np.random.choice(len(transitions), p=probs)
        _, s_next, reward, done = transitions[idx]

        self.state = self._to_rowcol(s_next)
        return s_next, reward, done, {}

    def render(self, values=None, policy=None):
        """Print a text rendering of the grid."""
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                if (r, c) == self.state:
                    cell = ' A '
                elif (r, c) == self.goal:
                    cell = ' G '
                elif (r, c) in self.walls:
                    cell = ' ■ '
                elif (r, c) in self.traps:
                    cell = ' X '
                elif policy is not None:
                    s = self._to_index(r, c)
                    cell = f' {self.ACTION_NAMES[policy[s]]} '
                elif values is not None:
                    s = self._to_index(r, c)
                    cell = f'{values[s]:+.1f}'
                else:
                    cell = ' . '
                row_str.append(cell)
            print('|'.join(row_str))
            if r < self.rows - 1:
                print('-' * (len(row_str) * 4 - 1))
        print()


class ChainMDP:
    """
    A simple N-state chain MDP for understanding Bellman equations.

    States: 0, 1, ..., n-1
    Actions: 0=LEFT, 1=RIGHT
    State 0 and n-1 are terminal with rewards left_reward and right_reward.
    """

    def __init__(self, n=5, left_reward=0.0, right_reward=1.0, step_reward=-0.01):
        self.n = n
        self.left_reward = left_reward
        self.right_reward = right_reward
        self.step_reward = step_reward
        self.n_states = n
        self.n_actions = 2
        self.state = n // 2  # Start in the middle

        self.P = {}
        for s in range(n):
            self.P[s] = {}
            for a in range(2):
                if s == 0 or s == n - 1:
                    self.P[s][a] = [(1.0, s, 0.0, True)]
                else:
                    s_next = s - 1 if a == 0 else s + 1
                    if s_next == 0:
                        reward = left_reward
                    elif s_next == n - 1:
                        reward = right_reward
                    else:
                        reward = step_reward
                    done = s_next == 0 or s_next == n - 1
                    self.P[s][a] = [(1.0, s_next, reward, done)]

    def reset(self):
        self.state = self.n // 2
        return self.state

    def step(self, action):
        transitions = self.P[self.state][action]
        _, s_next, reward, done = transitions[0]
        self.state = s_next
        return s_next, reward, done, {}
