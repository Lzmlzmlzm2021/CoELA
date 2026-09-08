"""Non-manipulating Box Scout built on CoELA's original visual memory."""

from lm_agent import lm_agent


class ScoutAgent(lm_agent):
    """A mobile RGB-D/GT-segmentation scout with no manipulation capability."""

    def __init__(self, agent_id, logger, max_frames, args,
                 output_dir="results"):
        super().__init__(agent_id, logger, max_frames, args, output_dir,
                         agent_role="scout")
        self.agent_type = "scout_agent"

    def reset(self, obs, goal_objects=None, output_dir=None, env_api=None,
              rooms_name=None, agent_color=[-1, -1, -1], agent_id=0,
              gt_mask=True, save_img=True):
        if not gt_mask:
            raise ValueError(
                "Box Scout currently supports the GT segmentation frontend only")
        return super().reset(
            obs=obs,
            goal_objects=goal_objects,
            output_dir=output_dir,
            env_api=env_api,
            rooms_name=rooms_name,
            agent_color=agent_color,
            agent_id=agent_id,
            gt_mask=gt_mask,
            save_img=save_img,
        )

    @staticmethod
    def _reject_manipulation(action_name):
        raise PermissionError(
            f"Box Scout has no hands or gripper and cannot {action_name}")

    def gograsp(self):
        return self._reject_manipulation("grasp")

    def putin(self):
        return self._reject_manipulation("put objects into containers")

    def goput(self):
        return self._reject_manipulation("transport or drop objects")


# Backward-compatible snake_case alias for scripts that mirror ``lm_agent``.
scout_agent = ScoutAgent
