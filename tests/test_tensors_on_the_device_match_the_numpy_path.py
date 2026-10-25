"""Tensors on the device match the numpy path.

TensorEnv hands an environment's observation, rewards and flags out as tensors on a
device, each the numpy path's to the bit: on a CUDA device through pinned host buffers
that take turns, so no step writes over a buffer a copy may still be reading, and in its
resident mode with the market block written on the device from the market's cells
copied there once. On the CPU these run wherever torch is installed; on a CUDA device
they run in the gpu stage on the owner's card, a stress run varying the policy's pace so
a buffer written again too soon would show, and a loop of 4,096 environments that never
waits for the card (Specification, Tensors on a device; ADR 0040).
"""

import copy
import random
import re
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import probatum
import probatum.torch
from probatum import ProbatumTypeError, ProbatumValueError
from test_an_agent_carries_the_spec_it_trained_with import momentum
from test_an_episode_runs_again_as_a_backtest_from_its_start import (HOUR, VENUE,
                                                                     universe, walked)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="no CUDA device; the gpu stage hands the card in")

SETTINGS = [
    dict(),
    dict(normalise=True),
    dict(features=momentum, fill=0.0, normalise=True),
    dict(features=momentum, fill=0.0, layout="dict"),
    dict(actions="orders"),
    dict(sample=1, actions="discrete"),
    dict(autoreset="disabled", episode_length=None),
]
IDS = ["candles", "normalised", "function", "dict", "orders", "sampled", "disabled"]


def holey(rows=300, seed=3):
    """universe's two perpetuals, ETHUSDT listed at bar 30 and printing nothing on bars
    50 to 55 and BTCUSDT trading nothing on bars 100 to 120, so a window can hide a
    slot, skip a bar or hold no volume to average."""
    rng   = np.random.default_rng(seed)
    times = 1_700_000_000_000_000_000 + np.arange(rows, dtype=np.int64) * HOUR
    btc   = walked(30_000.0, rows, rng)
    eth   = walked(2_000.0, rows, rng)

    btc[100:121, 4] = 0.0
    eth[:30]        = np.nan
    eth[50:56]      = np.nan

    pair = [probatum.Data(bars, symbol=name, market="linear", time=times)
            for name, bars in (("BTCUSDT", btc), ("ETHUSDT", eth))]
    return pair[0].join(pair[1])


def built(settings, envs=3, seed=2):
    """An environment of ``settings`` on `holey`'s bars, each built the same from the
    same seed."""
    given = {**dict(window=8, num_envs=envs, seed=seed, episode_length=30), **settings}
    return probatum.env(holey(), VENUE, **given)


def ran(settings, device, resident, steps=80, pace=None, tensors=True):
    """Each step's returns from the numpy path and from TensorEnv, kept to the end, so
    a later step that changed an earlier tensor would show; TensorEnv takes the actions
    as tensors on the device, or with ``tensors`` off as the numpy path's own, which
    never wait for the device, and ``pace`` is called before each step."""
    host  = built(settings)
    dev   = probatum.torch.TensorEnv(built(settings), device=device, resident=resident)
    space = host.action_space
    space.seed(3)

    seen = [(host.reset()[0], dev.reset()[0])]
    for step in range(steps):
        actions = space.sample()
        if pace is not None:
            pace(step)
        given = torch.as_tensor(actions, device=device) if tensors else actions
        seen.append((host.step(actions), dev.step(given)))

    return seen


def same(host, dev, device):
    """Whether ``dev``, a tensor or a dict of them on ``device``, holds ``host``'s dtype,
    shape and bytes."""
    if isinstance(host, dict):
        return host.keys() == dev.keys() and all(same(host[key], dev[key], device)
                                                 for key in host)

    held = dev.cpu().numpy()
    return (dev.device.type == device and held.dtype == host.dtype
            and held.shape == host.shape and held.tobytes() == host.tobytes())


def held(seen, device):
    """Whether every step's returns, taken from TensorEnv on ``device``, are the numpy
    path's."""
    for host, dev in seen:
        if isinstance(host, tuple):
            # the info stays as the environment gives it
            assert host[4].keys() == dev[4].keys()
            if not all(same(a, b, device) for a, b in zip(host[:4], dev[:4])):
                return False
        elif not same(host, dev, device):
            return False
    return True


@pytest.mark.parametrize("resident", [False, True], ids=["host", "resident"])
@pytest.mark.parametrize("settings", SETTINGS, ids=IDS)
def test_every_tensor_on_the_cpu_is_the_numpy_path_s_to_the_bit(settings, resident):
    assert held(ran(settings, "cpu", resident), "cpu")


def test_same_step_autoreset_hands_its_final_observations_out_as_the_numpy_path_does():
    seen = ran(dict(autoreset="same_step"), "cpu", False)

    assert held(seen, "cpu")

    ended = [(host[4], dev[4]) for host, dev in seen[1:] if "final_obs" in host[4]]
    assert ended
    for host, dev in ended:
        assert np.array_equal(host["_final_obs"], dev["_final_obs"])
        mask = host["_final_obs"]
        assert np.array_equal(np.stack(list(host["final_obs"][mask])),
                              np.stack(list(dev["final_obs"][mask])))


def test_what_tensor_env_cannot_wrap_is_refused_naming_the_fix():
    with pytest.raises(ProbatumTypeError, match="^env: an object of type object; wrap "
                                                "the environment probatum.env returns$"):
        probatum.torch.TensorEnv(object())

    with pytest.raises(ProbatumTypeError, match=r"^resident: 1; give True or False$"):
        probatum.torch.TensorEnv(built({}), resident=1)

    with pytest.raises(ProbatumValueError, match=re.escape(
            "resident: True under same_step autoreset, whose final observations are "
            "written on the host; give autoreset=\"next_step\" or \"disabled\"")):
        probatum.torch.TensorEnv(built(dict(autoreset="same_step")), resident=True)

    with pytest.raises(ProbatumValueError, match=re.escape(
            "resident: True on an environment recording transitions, whose observations "
            "are kept whole on the host; record them with resident=False")):
        probatum.torch.TensorEnv(built(dict(record="transitions")), resident=True)


def test_a_clone_of_an_environment_tensor_env_took_over_is_one_of_its_own():
    env = built({})
    probatum.torch.TensorEnv(env, resident=True)

    obs, _ = env.clone().reset()

    assert np.array_equal(obs, built({}).reset()[0])


@pytest.mark.parametrize("resident", [False, True], ids=["host", "resident"])
def test_a_copy_of_tensor_env_wraps_a_clone_of_its_environment(resident):
    wrap  = probatum.torch.TensorEnv(built({}), resident=resident)
    space = wrap.action_space
    space.seed(3)

    wrap.reset()
    wrap.step(space.sample())

    twin    = copy.deepcopy(wrap)
    actions = [space.sample() for _ in range(5)]

    # the copy steps first, so a buffer or an environment it shared would show
    ahead = [twin.step(actions[k]) for k in range(5)]
    for mine, theirs in zip([wrap.step(actions[k]) for k in range(5)], ahead):
        assert all(torch.equal(a, b) for a, b in zip(mine[:4], theirs[:4]))


def test_tensor_env_s_host_path_records_the_transitions_the_numpy_path_does():
    host  = built(dict(record="transitions"))
    dev   = probatum.torch.TensorEnv(built(dict(record="transitions")))
    space = host.action_space
    space.seed(3)

    host.reset()
    dev.reset()
    for _ in range(40):
        actions = space.sample()
        host.step(actions)
        dev.step(torch.as_tensor(actions))

    assert len(host.transitions) > 0
    for name in host.transitions.columns:
        assert dev.env.transitions[name].tobytes() == host.transitions[name].tobytes()


def test_tensor_env_carries_the_environment_s_spec_spaces_and_provenance():
    env  = built(dict(features=momentum, fill=0.0))
    wrap = probatum.torch.TensorEnv(env, resident=True)

    assert wrap.agent_spec == env.agent_spec and wrap.provenance is env.provenance
    assert wrap.single_observation_space == env.single_observation_space
    assert wrap.num_envs == 3 and wrap.device == torch.device("cpu")
    assert len(wrap.describe()) == len(env.describe())


def eight(data):
    """momentum four times over, eight features an instrument."""
    return np.concatenate([momentum(data)] * 4, axis=2)


def test_a_step_crosses_to_the_device_the_bytes_the_specification_counts():
    # the worked example: 4,096 environments of ten instruments, eight features and a
    # 64-bar window, a step's rewards and flags crossing beside its observation
    rng   = np.random.default_rng(0)
    times = 1_700_000_000_000_000_000 + np.arange(80, dtype=np.int64) * HOUR
    data  = probatum.Data(
        np.stack([walked(100.0, 80, rng) for _ in range(10)], axis=1),
        symbol = [f"C{i}" for i in range(10)],
        market = "linear",
        time   = times,
    )
    env   = probatum.env(data, probatum.Venue(), window=64, features=eight,
                         num_envs=4_096, seed=0)

    assert 4_096 * env.single_observation_space.shape[0] * 4 == 84_606_976

    wrap    = probatum.torch.TensorEnv(env, resident=True)
    crossed = []
    moved   = wrap._moved

    def counted(held):
        crossed.append(held.nbytes)
        return moved(held)

    wrap._moved = counted
    wrap.reset()

    assert sum(crossed) == 1_081_344

    crossed.clear()
    wrap.step(np.zeros(env.action_space.shape, dtype=np.float32))

    assert sum(crossed) == 1_081_344 + 4_096 * (8 + 2)


@CUDA
@pytest.mark.gpu
@pytest.mark.parametrize("resident", [False, True], ids=["host", "resident"])
@pytest.mark.parametrize("settings", SETTINGS, ids=IDS)
def test_every_tensor_on_the_card_is_the_numpy_path_s_however_the_policy_paces_it(
        settings, resident):
    rng = random.Random(f"pace {resident} {sorted(settings)}")

    def pace(step):
        # the card kept busy for a while, the host made to wait for a while, or neither,
        # so a buffer written again before its copy has run would show
        if rng.random() < 0.5:
            torch.cuda._sleep(rng.randrange(1, 4_000_000))
        if rng.random() < 0.3:
            time.sleep(rng.choice([0.0005, 0.002]))

    # actions from the host, which never make the host wait for the card, so the copies
    # queue behind the policy's work
    seen = ran(settings, "cuda", resident, steps=400, pace=pace, tensors=False)
    torch.cuda.synchronize()

    assert held(seen, "cuda")
    assert held(ran(settings, "cuda", resident, steps=40), "cuda")


@CUDA
@pytest.mark.gpu
def test_tensor_env_resets_the_environments_the_flags_it_handed_out_name():
    # the disabled mode leaves an ended episode for the caller to reset, by the flags
    # the card handed back
    settings = dict(autoreset="disabled", episode_length=3)
    host     = built(settings)
    dev      = probatum.torch.TensorEnv(built(settings), device="cuda", resident=True)
    space    = host.action_space
    space.seed(3)

    seen = [(host.reset()[0], dev.reset()[0])]
    for _ in range(9):
        actions = space.sample()
        mine, theirs = host.step(actions), dev.step(actions)
        seen.append((mine, theirs))
        if (mine[2] | mine[3]).any():
            seen.append((host.reset(options={"reset_mask": mine[2] | mine[3]})[0],
                         dev.reset(options={"reset_mask": theirs[2] | theirs[3]})[0]))

    torch.cuda.synchronize()

    assert held(seen, "cuda") and len(seen) == 13


@CUDA
@pytest.mark.gpu
def test_tensor_env_keeps_to_the_card_current_at_the_wrap():
    wrap = probatum.torch.TensorEnv(built({}), device="cuda", resident=True)
    card = torch.device("cuda", torch.cuda.current_device())

    obs, _ = wrap.reset()

    assert wrap.device == card and obs.device == card


@CUDA
@pytest.mark.gpu
@pytest.mark.parametrize("resident", [False, True], ids=["host", "resident"])
def test_a_loop_of_4096_environments_never_waits_for_the_card(resident):
    env   = probatum.torch.TensorEnv(
        built({}, envs=4_096),
        device   = "cuda",
        resident = resident,
    )
    space = env.action_space
    space.seed(1)

    obs, _ = env.reset()
    marks, ahead = [], []
    for _ in range(40):
        # the policy's work on the card, about 40 ms of it, queued before each step
        torch.cuda._sleep(60_000_000)
        obs, reward, terminated, truncated, _ = env.step(space.sample())

        # the card has not yet finished the step before this one, so the host waited
        # neither for it nor for this one's copies, two buffers taking turns
        if marks:
            ahead.append(not marks[-1].query())
        marks.append(torch.cuda.Event())
        marks[-1].record()

    torch.cuda.synchronize()

    assert all(ahead) and obs.shape == (4_096, 92) and obs.device.type == "cuda"
