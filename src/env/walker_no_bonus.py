import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class WalkerNoBonusEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
        External forces (sim.data.cfrc_ext) are removed from the observation, and survive_reward = 0.
        Otherwise identical to Walker2d-v2 from
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/walker2d.py
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125,
    }
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'walker2d.xml', 5)
        utils.EzPickle.__init__(self)
        self._max_episode_steps = 1000

    def step(self, a):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter, height, ang = self.sim.data.qpos[0:3]

        #alive_bonus = 1.0
        survive_reward = 0.0
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost= 1e-3 * np.square(a).sum()
        reward = forward_reward - ctrl_cost + survive_reward
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        #done = not not_done
        ob = self._get_obs()
        info = dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            #reward_contact=-contact_cost,
            reward_survive=survive_reward,
            violation=done
        )
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def check_done(self, states):
        return self.check_violation(states)

    def check_violation(self, states):
        height = states[:,0]
        ang = states[:,1]
        return ~((height > 0.8) & (height < 2.0) & (ang > -1.0) & (ang < 1.0))
