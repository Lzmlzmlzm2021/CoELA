import argparse
import os
import json
import gym
import time
import pickle
import logging
import sys

# add this dictionary to python env path:
base_path = os.getcwd()
sys.path.append(base_path)

from h_agent import H_agent
from eval_artifacts import (
    HUMAN_BOX_PROTOCOL,
    LEGACY_PROTOCOL,
    SUPPORTED_PROTOCOLS,
    atomic_write_json,
    build_aggregate_result,
    build_episode_result,
    load_episode_result,
)

gym.envs.registration.register(
    id='transport_challenge_MA',
    entry_point='tdw_gym:TDW'
)

class Challenge:
    def __init__(self, logger, port, data_path, output_dir, number_of_agents = 2, max_frames = 3000, launch_build = True, screen_size = 512, data_prefix = 'dataset/nips_dataset/', gt_mask = True, save_img = True, embodiments = None, result_protocol = LEGACY_PROTOCOL):
        self.env = gym.make("transport_challenge_MA", port = port, number_of_agents = number_of_agents, save_dir = output_dir, max_frames = max_frames, launch_build = launch_build, screen_size = screen_size, data_prefix = data_prefix, gt_mask = gt_mask, embodiments = embodiments)
        self.gt_mask = gt_mask
        self.logger = logger
        self.logger.debug(port)
        self.logger.info("Environment Created")
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.save_img = save_img
        self.data = json.load(open(os.path.join(data_prefix, data_path), "r"))
        if result_protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"unsupported result protocol: {result_protocol!r}")
        self.result_protocol = result_protocol
        self.logger.info("done")

    def submit(self, agents, logger, eval_episodes, coordinator=None):
        total_finish = 0.0
        if eval_episodes[0] == -1:
            eval_episodes = list(range(len(self.data)))
        else:
            eval_episodes = list(eval_episodes)
        num_eval_episodes = len(eval_episodes)

        start = time.time()
        results = {}
        for i, episode in enumerate(eval_episodes):
            start_time = time.time()
            result_path = os.path.join(
                self.output_dir, str(episode), 'result_episode.json')
            if os.path.exists(result_path):
                result = load_episode_result(
                    result_path,
                    episode,
                    self.data[episode],
                    self.max_frames,
                    protocol=self.result_protocol,
                )
                self.logger.info(
                    "Resume validated completed episode %s using protocol %s",
                    episode,
                    self.result_protocol,
                )
                total_finish += result['finish'] / result['total']
                results[episode] = result
                continue
            # The episode has been evaluated before

            if not os.path.exists(os.path.join(self.output_dir, str(episode))):
                os.makedirs(os.path.join(self.output_dir, str(episode)))
            self.logger.info('Episode {} ({}/{})'.format(episode, i + 1, num_eval_episodes))
            self.logger.info(f"Resetting Environment ... data is {self.data[episode]}")
            state, info, env_api = self.env.reset(seed=self.data[episode]['seed'], options=self.data[episode], output_dir = os.path.join(self.output_dir, str(episode)))
            for id, agent in enumerate(agents):
                if type(env_api) == list:
                    curr_api = env_api[id]
                else: curr_api = env_api
                if info['goal_description'] is not None:
                    if agent.agent_type == 'h_agent':
                        agent.reset(goal_objects = info['goal_description'], output_dir = os.path.join(self.output_dir, str(episode)), env_api = curr_api, agent_color = info['agent_colors'][id], agent_id = id, gt_mask = self.gt_mask, save_img = self.save_img)
                    elif agent.agent_type in ('lm_agent', 'scout_agent'):
                        agent.reset(obs = state[str(id)], goal_objects = info['goal_description'], output_dir = os.path.join(self.output_dir, str(episode)), env_api = curr_api, agent_color = info['agent_colors'][id], agent_id = id, rooms_name=info['rooms_name'], gt_mask = self.gt_mask, save_img = self.save_img)
                    else:
                        raise Exception(f"{agent.agent_type} not available")
                else:
                    agent.reset(output_dir = os.path.join(self.output_dir, str(episode)))
            if coordinator is not None:
                coordinator.reset(
                    info['goal_description'], episode_id=episode)
            self.logger.info(f"Environment Reset. Took {time.time() - start_time} secs")
            local_finish = self.env.check_goal()
            done = False
            step_num = 0
            local_reward = 0.0
            while not done:
                step_num += 1
                actions = {}
                if self.save_img: self.env.save_images(os.path.join(self.output_dir, str(episode), 'Images'))
                if coordinator is not None:
                    actions = coordinator.act(
                        state,
                        delivered_objects=(
                            self.env.unwrapped.get_delivered_objects()),
                    )
                else:
                    for agent_id, agent in enumerate(agents):
                        actions[str(agent_id)] = agent.act(state[str(agent_id)])
                state, reward, done, info = self.env.step(actions)
                local_reward += reward
                local_finish = self.env.check_goal()
                if coordinator is not None:
                    coordinator.observe_outcome(
                        state,
                        local_finish,
                        info,
                        delivered_objects=(
                            self.env.unwrapped.get_delivered_objects()),
                    )
                self.logger.info(f"Executing step {step_num} for episode: {episode}, actions: {actions}, finish: {local_finish}, frame: {self.env.num_frames}")
                if done:
                    break
            if coordinator is not None:
                coordinator.finalize(local_finish)
            total_finish += local_finish[0] / local_finish[1]
            result = build_episode_result(
                episode,
                self.data[episode],
                self.max_frames,
                local_finish[0],
                local_finish[1],
                protocol=self.result_protocol,
            )
            atomic_write_json(result_path, result)
            results[episode] = result
        avg_finish = total_finish / num_eval_episodes
        results = build_aggregate_result(
            results,
            avg_finish,
            protocol=self.result_protocol,
        )
        atomic_write_json(
            os.path.join(self.output_dir, 'eval_result.json'),
            results,
            indent=4,
        )
        self.logger.info(f'eval done, avg transport rate {avg_finish}')
        self.logger.info('time: {}'.format(time.time() - start))
        return avg_finish

    def close(self):
        self.env.close()

def init_logs(output_dir, name = 'simple_example'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, "output.log"))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--experiment_name", type = str, default = "try")
    parser.add_argument("--run_id", type=str, default='run_0')
    parser.add_argument("--data_path", type=str, default="test_env.json")
    parser.add_argument("--data_prefix", type=str, default="dataset/dataset_train/")
    parser.add_argument("--port", default=1071, type=int)
    parser.add_argument("--agents", nargs='+', type=str, default=("h_agent",))
    parser.add_argument("--embodiments", nargs='+', type=str, default=None,
                        help="physical embodiment for each agent: replicant or box")
    parser.add_argument("--eval_episodes", nargs='+', default=(-1,), type=int, help="which episodes to evaluate on")
    parser.add_argument("--max_frames", default=3000, type=int, help="max frames per episode")
    parser.add_argument("--no_launch_build", action='store_true')
    parser.add_argument("--communication", action='store_true')
    parser.add_argument(
        "--peer_consult", action='store_true',
        help=("enable the evidence-centric Memory Board, short-lived intents, "
              "container-aware transport evidence, selective peer review "
              "and conflict checking for two Replicants"),
    )
    parser.add_argument(
        "--peer_consult_policy", choices=("llm", "legacy"), default="llm",
        help="PeerConsultV4: use the shared LLM-owned core (default), or reproduce the historical V4 policy",
    )
    parser.add_argument(
        "--peer_review_mode", choices=("deterministic",),
        default="deterministic",
    )
    parser.add_argument(
        "--peer_consult_protocol",
        choices=("PeerConsultV3", "PeerConsultV3.1", "PeerConsultV3.2",
                 "PeerConsultV3.3", "PeerConsultV3.4",
                 "PeerConsultV3.5", "PeerConsultV4"),
        default="PeerConsultV3",
        help=("PeerConsultV3.1 keeps the V3 Memory Board but gates natural "
              "language on physical events, debounces room reassignment, "
              "and prioritizes delivery before the frame budget is unsafe; "
              "PeerConsultV3.2 additionally reconciles planner delivery "
              "memory with evaluator evidence and commits bankable payloads; "
              "PeerConsultV3.3 adds persistent task scheduling, decision-first "
              "communication, coverage exploration and evaluator-confirmed "
              "safe delivery without the V3.2 direct-drop override; "
              "PeerConsultV3.4 reconciles shared delivery knowledge, payload "
              "identity, parked containers and persistent perception events; "
              "PeerConsultV3.5 retains those semantics while adding bounded "
              "plan-progress recovery and episode-isolated shared evidence; "
              "PeerConsultV4 keeps the original CoELA executor while adding "
              "complete legal candidates, persistent tasks, atomic factual "
              "coordination; --peer_consult_policy selects common LLM policy or historical guards"),
    )
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--no_gt_mask", action='store_true')
    # LLM parameters
    parser.add_argument('--source', default='openai',
        choices=['hf', 'openai'],
        help='openai API or load huggingface models')
    parser.add_argument('--lm_id', default='gpt-3.5-turbo',
                        help='name for openai engine or huggingface model name/path')
    parser.add_argument('--prompt_template_path', default='LLM/prompt_single.csv',
                        help='path to prompt template file')
    parser.add_argument("--t", default=0.7, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--max_tokens", default=64, type=int)
    parser.add_argument("--n", default=1, type=int)
    parser.add_argument("--logprobs", default=1, type=int)
    parser.add_argument("--cot", action='store_true', help="use chain-of-thought prompt")
    parser.add_argument("--echo", action='store_true', help="to include prompt in the outputs")
    parser.add_argument("--screen_size", default=512, type=int)
    parser.add_argument("--no_save_img", action='store_true', help="do not save images", default=False)
    parser.add_argument(
        "--result_protocol",
        choices=sorted(SUPPORTED_PROTOCOLS),
        default=LEGACY_PROTOCOL,
        help=("artifact/resume protocol; human_box_v2 binds completed JSON "
              "to the exact dataset episode and frame limit"),
    )
    args = parser.parse_args()

    args.number_of_agents = len(args.agents)
    if args.embodiments is None:
        args.embodiments = ["replicant"] * args.number_of_agents
    if len(args.embodiments) != args.number_of_agents:
        parser.error("--embodiments must have exactly one value per --agents entry")
    invalid_embodiments = [x for x in args.embodiments if x not in ("replicant", "box")]
    if invalid_embodiments:
        parser.error(f"unsupported embodiment(s): {invalid_embodiments}")
    for index, (agent, embodiment) in enumerate(zip(args.agents, args.embodiments)):
        if agent == "scout_agent" and embodiment != "box":
            parser.error(f"agent {index}: scout_agent requires the box embodiment")
        if embodiment == "box" and agent != "scout_agent":
            parser.error(f"agent {index}: the box embodiment requires scout_agent")
    if args.peer_consult:
        if args.agents != ["lm_agent", "lm_agent"]:
            parser.error("--peer_consult requires --agents lm_agent lm_agent")
        if args.embodiments != ["replicant", "replicant"]:
            parser.error("--peer_consult requires two replicant embodiments")
        if not args.communication:
            parser.error("--peer_consult requires --communication so planners "
                         "receive the public Memory Board")
    os.makedirs(args.output_dir, exist_ok = True)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(args.output_dir, exist_ok = True)
    args.output_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(args.output_dir, exist_ok = True)
    logger = init_logs(args.output_dir)

    if args.result_protocol == HUMAN_BOX_PROTOCOL:
        if args.agents != ["lm_agent", "scout_agent"]:
            parser.error("human_box_v2 requires --agents lm_agent scout_agent")
        if args.embodiments != ["replicant", "box"]:
            parser.error("human_box_v2 requires --embodiments replicant box")
    challenge = Challenge(logger, args.port, args.data_path, args.output_dir, args.number_of_agents, args.max_frames, not args.no_launch_build, screen_size = args.screen_size, data_prefix=args.data_prefix, gt_mask = not args.no_gt_mask, save_img = not args.no_save_img, embodiments=args.embodiments, result_protocol=args.result_protocol)
    agents = []
    for i, agent in enumerate(args.agents):
        if agent == 'h_agent':
            agents.append(H_agent(i, logger, args.max_frames, args.output_dir))
        elif agent == 'lm_agent':
            # Import lazily so the heuristic/GT-mask smoke test doesn't require
            # any LLM-only dependencies.
            from lm_agent import lm_agent
            agents.append(lm_agent(i, logger, args.max_frames, args, args.output_dir))
        elif agent == 'scout_agent':
            from scout_agent import ScoutAgent
            agents.append(ScoutAgent(i, logger, args.max_frames, args, args.output_dir))
        else:
            raise ValueError(f"Unknown agent type: {agent}")
    coordinator = None
    if args.peer_consult:
        from peer_consult import TDWPeerConsultCoordinator
        coordinator_class = TDWPeerConsultCoordinator
        if args.peer_consult_protocol == "PeerConsultV4" and args.peer_consult_policy == "llm":
            from peer_consult_common import CommonTDWPeerConsultCoordinator
            coordinator_class = CommonTDWPeerConsultCoordinator
        coordinator = coordinator_class(
            agents=agents,
            logger=logger,
            output_dir=args.output_dir,
            review_mode=args.peer_review_mode,
            max_frames=args.max_frames,
            protocol_version=args.peer_consult_protocol,
        )
    try:
        challenge.submit(agents, logger, args.eval_episodes,
                         coordinator=coordinator)
    finally:
        challenge.close()

if __name__ == "__main__":
    main()
