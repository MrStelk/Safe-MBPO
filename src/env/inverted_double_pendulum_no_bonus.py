import numpy as np

from gym import utils
#from gym.envs.mujoco import MuJocoPyEnv
from gym.spaces import Box
from gym.envs.mujoco import mujoco_env


class InvertedDoublePendulumNoBonusEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self):
        #observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_double_pendulum.xml', 5)
        utils.EzPickle.__init__(self)
        self._max_episode_steps = 1000

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        ob = self._get_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        #alive_bonus = 10
        alive_bonus = 0.0
        r = alive_bonus - dist_penalty - vel_penalty
        done = bool(y <= 1)

        #if self.render_mode == "human":
        #    self.render()
        info = dict(
            reward_dist=-dist_penalty,
            reward_ctrl=-vel_penalty,
            #reward_contact=-contact_cost,
            reward_survive=alive_bonus,
            violation=done
        )
        return ob, r, done, info

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos[:1],  # cart x pos
                np.sin(self.sim.data.qpos[1:]),  # link angles
                np.cos(self.sim.data.qpos[1:]),
                np.clip(self.sim.data.qvel, -10, 10),
                np.clip(self.sim.data.qfrc_constraint, -10, 10),
            ]
        ).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1,
        )
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]
      
    def check_done(self, states):
        return self.check_violation(states)

    def check_violation(self, states):
        #height = states[:,0]
        #ang = states[:,1]
        _, _, y = self.sim.data.site_xpos[0]   # directly from simulator
        return bool(y <= 1)
