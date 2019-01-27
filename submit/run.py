import argparse
import path
import os
import copy
import torch
import numpy as np
from catalyst.utils.config import parse_args_uargs
from catalyst.utils.misc import set_global_seeds, import_module, create_if_need
from envs.prosthetics_preprocess import \
    preprocess_obs_round2 as preprocess_state
from osim.env import ProstheticsEnv
import atexit
import multiprocessing as mp


os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
set_global_seeds(42)
golden_seeds = np.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "golden_seeds.npz"))["seeds"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlgoWrapper:
    def __init__(self, actor, critics, history_len, consensus="min"):
        self.actor = actor.to(DEVICE)
        self.critics = [x.to(DEVICE) for x in critics]
        self.history_len = history_len
        self.consensus = consensus

    @staticmethod
    def _act(actor, state):
        with torch.no_grad():
            states = torch.Tensor(state).unsqueeze(0).to(DEVICE)
            action = actor(states)
            action = action[0].detach().cpu().numpy()
            return action

    @staticmethod
    def _evaluate(critic, state, action):
        with torch.no_grad():
            states = torch.Tensor(state).unsqueeze(0).to(DEVICE)
            actions = torch.Tensor(action).unsqueeze(0).to(DEVICE)
            value = critic(states, actions)
            value = value[0].detach().cpu().numpy()
            value = np.mean(value)
            return value

    def get_state(self, buffer):
        return buffer.get_state(history_len=self.history_len)

    def act(self, state):
        return self._act(self.actor, state)

    def evaluate(self, state, action):
        values = [self._evaluate(c, state, action) for c in self.critics]
        value = np.__dict__[self.consensus](values)
        return value


class SamplerBuffer:
    def __init__(self, capacity, state_shape):
        self.size = capacity
        self.state_shape = state_shape
        self.pointer = 0
        self.states = np.empty(
            (self.size,) + tuple(self.state_shape), dtype=np.float32)

    def init_with_state(self, state):
        self.states[0] = state
        self.pointer = 0

    def get_state(self, history_len=1, pointer=None):
        pointer = pointer or self.pointer
        state = np.zeros(
            (history_len,) + self.state_shape, dtype=np.float32)
        indices = np.arange(
            max(0, pointer - history_len + 1), pointer + 1)
        state[-len(indices):] = self.states[indices]
        return state

    def push_transition(self, s_tp1):
        """ transition = [s_tp1, a_t, r_t, d_t, ts_t]
        """
        self.states[self.pointer + 1] = s_tp1
        self.pointer += 1

    def get_states_history(self, history_len=1):
        states = [
            self.get_state(history_len=history_len, pointer=i)
            for i in range(self.pointer)]
        states = np.array(states)
        return states


def get_actor_weights(actor, exclude_norm=False):
    state_dict = actor.state_dict()
    if exclude_norm:
        state_dict = {
            key: value for key, value in state_dict.items()
            if all(x not in key for x in ["norm", "lstm"])
        }
    state_dict = {
        key: value.clone()
        for key, value in state_dict.items()
    }
    return state_dict


def set_actor_weights(actor, weights, strict=True):
    actor.load_state_dict(weights, strict=strict)


def set_params_noise(
        actor, states,
        target_d=0.5, tol=1e-3, max_steps=1000):
    exclude_norm = True
    orig_weights = get_actor_weights(actor, exclude_norm=exclude_norm)
    orig_act = actor(states)

    sigma_min = 0.
    sigma_max = 100.
    sigma = sigma_max
    step = 0

    while step < max_steps:
        dist = torch.distributions.normal.Normal(0, sigma)
        weights = {
            key: w.clone() + dist.sample(w.shape)
            for key, w in orig_weights.items()
        }
        set_actor_weights(actor, weights, strict=not exclude_norm)
        new_act = actor(states)

        diff = new_act - orig_act
        d = torch.mean(torch.sqrt(torch.sum(torch.pow(diff, 2), 1))).item()

        dd = d - target_d
        if np.abs(dd) < tol:
            break

        # too big sigma
        if dd > 0:
            sigma_max = sigma
        # too small sigma
        else:
            sigma_min = sigma
        sigma = sigma_min + (sigma_max - sigma_min) / 2
        step += 1
    return d


parser = argparse.ArgumentParser()
parser.add_argument("--visualize", action="store_true", default=False)
parser.add_argument("--print", action="store_true", default=False)

parser.add_argument("--logdir", type=str, default=None)
parser.add_argument("--logdir-start", type=str, default=None)
parser.add_argument("--logdir-run", type=str, default=None)
parser.add_argument("--logdir-side", type=str, default=None)

parser.add_argument("--action-noise", type=int, default=0)
parser.add_argument("--param-noise", type=int, default=0)
parser.add_argument("--time-trick", type=float, default=-1)   # used: 0.5
parser.add_argument("--action-mixin", type=int, default=1)    # used: 3
parser.add_argument("--change-trick", type=int, default=0)    # used: 50
parser.add_argument("--side-trick", type=float, default=-1)   # used: 1.0

parser.add_argument("--in-consensus", type=str, default="min")
parser.add_argument("--out-consensus", type=str, default="mean")

parser.add_argument("--outdir", type=str, required=True)
parser.add_argument("--seeds", type=str, default=None)
parser.add_argument("--n-cpu", type=int, default=1)
args = parser.parse_args()

if args.seeds is not None:
    SEEDS = list(map(int, args.seeds.split(",")))
else:
    SEEDS = golden_seeds

LOGDIR = args.logdir
LOGDIR_START = args.logdir_start
LOGDIR_RUN = args.logdir_run
LOGDIR_SIDE = args.logdir_side

VIS = args.visualize
ACCURACY = 1e-5
PRINT = args.print

TIME_TRICK = args.time_trick
CHANGE_TRICK = args.change_trick
ACTION_MIXIN = args.action_mixin
assert ACTION_MIXIN > 0
ACTION_INTERVAL = 1. / ACTION_MIXIN
ACTION_DELTAS = np.arange(0, 1 + ACTION_INTERVAL, ACTION_INTERVAL)
ACTION_NOISE = args.action_noise
PARAM_NOISE = args.param_noise
SIDE_TRICK = args.side_trick

IN_CONSENSUS = args.in_consensus
OUT_CONSENSUS = args.out_consensus

NCPU = args.n_cpu
OUTDIR = args.outdir
create_if_need(OUTDIR)

assert LOGDIR is not None \
       or (LOGDIR_START is not None and LOGDIR_RUN is not None)
LOGDIR_START = LOGDIR_START or LOGDIR


def algos_by_dir(dir):
    algos = []
    dirs = path.Path(dir).listdir()
    for logpath in dirs:
        config_path = logpath + "/config.json"
        checkpoints = path.Path(logpath).glob("*.pth.tar")
        for checkpoint_path in checkpoints:
            args = argparse.Namespace(config=config_path)
            args, config = parse_args_uargs(args, [])
            config.get("algorithm", {}).pop("resume", None)
            config.get("algorithm", {}).pop("load_optimizer", None)

            algo_module = import_module("algo_module", args.algorithm)
            trainer_kwargs = algo_module.prepare_for_trainer(config)

            algorithm = trainer_kwargs["algorithm"]
            algorithm.load_checkpoint(checkpoint_path, load_optimizer=False)

            actor_ = algorithm.actor.eval()

            name = str(algorithm.__class__).lower()
            if "ensemblecritic" in name:
                critics_ = [x.eval() for x in algorithm.critics]
            elif "td3" in name:
                critics_ = [algorithm.critic.eval(), algorithm.critic2.eval()]
            else:
                raise NotImplemented

            history_len = trainer_kwargs["history_len"]

            algos.append(
                AlgoWrapper(
                    actor=actor_,
                    critics=critics_,
                    history_len=history_len,
                    consensus=IN_CONSENSUS))
    return algos


ALGO_START = algos_by_dir(LOGDIR_START) if LOGDIR_START is not None else None
ALGO_RUN = algos_by_dir(LOGDIR_RUN) if LOGDIR_RUN is not None else None
ALGO_SIDE = algos_by_dir(LOGDIR_SIDE) if LOGDIR_SIDE is not None else None


action_mean = .5
action_std = .5
preprocess_action = lambda x: x * action_std + action_mean


def run_episodes(algo_start, algo_run, seeds, algo_side=None):
    try:
        seed = seeds.pop()
    except:
        return
    max_ep_length = int(1e3)
    algo_side = algo_side or algo_run
    current_algo_start = [copy.deepcopy(x) for x in algo_start]
    current_algo_run = [copy.deepcopy(x) for x in algo_run]
    current_algo_side = [copy.deepcopy(x) for x in algo_side]

    env = ProstheticsEnv(
        visualize=VIS, integrator_accuracy=ACCURACY, seed=seed)
    env.change_model(
        model="3D", prosthetic=True, difficulty=1,
        seed=seed)
    state_desc = None
    reward_, step_, episode_ = 0, 0, 0
    reward_l = []
    buffer = SamplerBuffer(
        capacity=max_ep_length + 42,
        state_shape=(344, ))

    while True:
        if state_desc is None:
            state_desc = env.reset(project=False)
            if not state_desc:
                break
            state = preprocess_state(state_desc)
            state.append(-1.0)
            buffer.init_with_state(state)
        else:
            state = preprocess_state(state_desc)
            time_pass = step_ / max_ep_length
            if TIME_TRICK > 0:
                state.append(min((time_pass - 0.5) * 2., TIME_TRICK))
            else:
                state.append((time_pass - 0.5) * 2.)
            buffer.push_transition(state)

        algos = current_algo_run \
            if current_algo_run is not None and step_ > CHANGE_TRICK \
            else current_algo_start
        if PRINT:
            print(f"STEP: {step_}\tTARGET: {state[-4:-1]}")
        if 0 < SIDE_TRICK < np.abs(state[-2]): #and step_ < 800:
            if PRINT:
                print(f"GOING SIDE: {step_} {state[-2]}")
            algos = current_algo_side

        states = [x.get_state(buffer) for x in algos]
        actions = [a.act(s) for a, s in zip(algos, states)]
        if len(actions) > 1 or ACTION_NOISE > 0:

            if len(actions) % 2 == 0:
                mid = len(actions)//2
                new_actions = []
                for a1, a2 in zip(actions[:mid], actions[mid:]):
                    actions = [
                        delta * a1 + (1 - delta) * a2
                        for delta in ACTION_DELTAS]
                    new_actions.extend(actions)
                actions = new_actions

            if ACTION_NOISE > 0:
                noise_actions = [
                    x + np.clip(np.random.normal(0.0, 0.009, (19, )), -0.1, 0.1)
                    for x in actions for _ in range(ACTION_NOISE)
                ]
                actions.extend(noise_actions)
            values = [
                np.__dict__[OUT_CONSENSUS]([
                    c.evaluate(s, a)
                    for s, c in zip(states, algos)])
                for a in actions]
            best_action_i = np.argmax(values)
        else:
            best_action_i = 0

        best_action = actions[best_action_i]
        best_action = preprocess_action(best_action)

        try:
            state_desc, reward, done, _ = env.step(best_action, project=False)
        except IndexError:
            done = True
            reward = 0
        reward_ += reward
        step_ += 1
        reward_l.append(reward)
        if done:
            if LOGDIR_RUN is not None:
                print(
                    f"REWARD: {reward_}\t"
                    f"STEPS: {step_}\t"
                    f"SEED: {seed}\t"
                    f"TIME: {TIME_TRICK}\t"
                    f"RUN at: {CHANGE_TRICK}")
            else:
                print(
                    f"REWARD: {reward_}\t"
                    f"STEPS: {step_}\t"
                    f"SEED: {seed}\t"
                    f"TIME: {TIME_TRICK}")
            np.savez(
                f"{OUTDIR}/ep_{seed}.npz",
                r=np.array(reward_l, dtype=np.float32))
            try:
                seed = seeds.pop()
            except:
                return

            if PARAM_NOISE > 0:
                current_algo_run = []
                for algo in algo_run:
                    current_algo_run.append(copy.deepcopy(algo))
                    states = buffer.get_states_history(
                        history_len=algo.history_len)
                    states = torch.Tensor(states).detach()
                    for _ in range(PARAM_NOISE):
                        new_actor = copy.deepcopy(algo.actor)
                        param_noise_d = set_params_noise(
                            actor=new_actor,
                            states=states)
                        print(f"NOISE D: {param_noise_d}")
                        new_algo = AlgoWrapper(
                            actor=new_actor,
                            critics=algo.critics,
                            history_len=algo.history_len)
                        current_algo_run.append(new_algo)

            env = ProstheticsEnv(
                visualize=VIS, integrator_accuracy=ACCURACY, seed=seed)
            env.change_model(
                model="3D", prosthetic=True, difficulty=1,
                seed=seed)
            state_desc = None
            reward_, step_, episode_ = 0, 0, 0
            reward_l = []
            buffer = SamplerBuffer(
                capacity=max_ep_length + 42,
                state_shape=(344,))


tasks = [
    {
        "algo_start": ALGO_START,
        "algo_run": ALGO_RUN,
        "algo_side": ALGO_SIDE,
        "seeds": seeds_.tolist()
    }
    for seeds_ in np.array_split(SEEDS, NCPU)]

processes = []


def on_exit():
    for p in processes:
        p.terminate()


atexit.register(on_exit)

for task in tasks:
    p = mp.Process(
        target=run_episodes,
        kwargs=dict(
            algo_start=task["algo_start"],
            algo_run=task["algo_run"],
            algo_side=task["algo_side"],
            seeds=task["seeds"]))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
