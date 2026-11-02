from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from brawl.reward.match_objective import MatchObjective


def event(**kwargs):
    return dict(
        op_stock_lost_step=0, self_stock_lost_step=0,
        op_stocks_left=3, self_stocks_left=3,
        op_delta_damage=0, self_delta_damage=0, damage_valid=True,
    ) | kwargs


def test_damage_is_capped_per_stock_and_resets_after_stock_loss():
    reward = MatchObjective()
    assert reward.step(event(op_delta_damage=80))["total_reward"] == pytest.approx(.8)
    assert reward.step(event(op_delta_damage=80))["total_reward"] == pytest.approx(.2)
    assert reward.step(event(op_delta_damage=50))["total_reward"] == 0
    assert reward.step(event(op_stock_lost_step=1, op_stocks_left=2))["total_reward"] == 5
    assert reward.step(event(op_delta_damage=20))["total_reward"] == pytest.approx(.2)


def test_terminal_bonus_once_and_no_passive_or_invalid_damage_reward():
    reward = MatchObjective()
    assert reward.step(event())["total_reward"] == 0
    assert reward.step(event(op_delta_damage=90, damage_valid=False))["total_reward"] == 0
    assert reward.step(event(op_stock_lost_step=1, op_stocks_left=0))["total_reward"] == 25
    assert reward.step(event(op_stock_lost_step=1, op_stocks_left=0))["total_reward"] == 0


def test_simultaneous_terminal_is_draw_and_respawn_does_not_pay_damage():
    reward = MatchObjective()
    result = reward.step(event(op_stock_lost_step=1, self_stock_lost_step=1,
                              op_stocks_left=0, self_stocks_left=0,
                              op_delta_damage=100, self_delta_damage=100))
    assert result["total_reward"] == 0
    assert result["game_win"] == result["game_loss"] == 0


def test_received_damage_cap_and_loss():
    reward = MatchObjective()
    assert reward.step(event(self_delta_damage=150))["total_reward"] == -.5
    assert reward.step(event(self_delta_damage=25))["total_reward"] == 0
    assert reward.step(event(self_stock_lost_step=1, self_stocks_left=0))["total_reward"] == -25


def test_nonfinite_damage_is_not_a_reward():
    reward = MatchObjective()
    assert reward.step(event(op_delta_damage=float("nan")))["total_reward"] == 0


def test_final_ppo_defaults_are_the_approved_small_updates():
    from brawl.pipelines.final_agent.config import parse_args
    cfg = parse_args(["--init-from", "skill.zip", "--world-replay", "replay"])
    assert cfg.learning_rate == 1e-4
    assert cfg.n_steps == 512 and cfg.batch_size == 256
    assert cfg.n_epochs == 4 and cfg.target_kl == .01
    assert cfg.gamma == .999 and cfg.clip_range == .1 and cfg.ent_coef == .01
    assert cfg.world_replay == "replay"


def test_match_wrapper_continues_stock_loss_and_truncates_without_loss(tmp_path):
    import gymnasium as gym
    from brawl.env.match import MatchTrainingEnv

    class Fake(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (3,))
        action_space = gym.spaces.Discrete(27)
        input_controller = SimpleNamespace(reset=lambda: None)
        def reset(self, **kwargs):
            return np.zeros(3), {}
        def step(self, action):
            return np.zeros(3), 999, False, False, event(
                self_stock_lost_step=1, self_stocks_left=2)

    env = MatchTrainingEnv(Fake(), max_steps=2, matches_path=tmp_path / "matches.csv")
    env.reset()
    _, reward, term, trunc, info = env.step(0)
    assert reward == -5 and not term and not trunc
    _, _, term, trunc, info = env.step(0)
    assert not term and trunc
    assert info["match_outcome"] == "timeout"
    assert info["reward_breakdown"]["game_loss"] == 0
    env.close()


def test_training_room_survives_more_than_three_losses_without_match_bonus(tmp_path):
    import gymnasium as gym
    from brawl.env.match import MatchTrainingEnv
    from brawl.state.state_spec import StateSpec

    class Arena(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (146,), dtype=np.float32)
        action_space = gym.spaces.Discrete(27)
        input_controller = SimpleNamespace(reset=lambda: None)
        stocks_health_provider = SimpleNamespace(self_stocks_left=3., op_stocks_left=3.)
        memory = SimpleNamespace(
            player=SimpleNamespace(exists=True, stocks=3),
            opponent=SimpleNamespace(exists=True, stocks=3),
            player_respawn_timer=0, opponent_respawn_timer=0,
            self_stocks_left=3., op_stocks_left=3.)
        resets = 0
        def reset(self, **kwargs):
            self.resets += 1
            return np.zeros(146, dtype=np.float32), {}
        def step(self, action):
            self.memory.self_stocks_left -= 1
            return np.zeros(146, dtype=np.float32), 0., False, False, event(
                self_stock_lost_step=1, self_stocks_left=self.memory.self_stocks_left)

    base = Arena()
    env = MatchTrainingEnv(base, mode="training_room", max_steps=2)
    env.reset()
    for _ in range(8):
        obs, reward, term, trunc, info = env.step(0)
        assert reward == -5 and not term
        assert obs[StateSpec.index("self_stocks_norm")] == 1.
        if trunc:
            env.reset()
    assert base.resets == 1


def test_real_ppo_rollouts_evaluate_save_and_record_neutral_update_gaps(tmp_path):
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv
    from brawl.pipelines.final_agent.campaign import FinalCampaign
    from brawl.pipelines.final_agent.config import parse_args
    from brawl.pipelines.final_agent.policy import build_final_model
    from brawl.env.match import MatchTrainingEnv, ReplayGymWrapper
    from brawl.training.checkpoints import make_checkpoint_metadata, FINAL_AGENT
    from brawl.world_model.replay_log import WorldReplayWriter, read_session

    class Arena(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (146,), dtype=np.float32)
        action_space = gym.spaces.Discrete(27)
        releases = 0
        @property
        def input_controller(self):
            return SimpleNamespace(reset=self.release)
        def release(self):
            self.releases += 1
        def reset(self, **kwargs):
            return np.zeros(146, dtype=np.float32), {}
        def step(self, action):
            return np.ones(146, dtype=np.float32) * .01, 0., False, False, event(op_delta_damage=1)

    cfg = parse_args(["--init-from", "unused.zip", "--n-steps", "8", "--batch-size", "4",
                      "--eval-every", "16", "--eval-episodes", "2", "--checkpoint-every", "8"])
    base = Arena()
    env = MatchTrainingEnv(base, mode="training_room", max_steps=4,
                           matches_path=tmp_path / "windows.csv")
    writer = WorldReplayWriter(tmp_path / "replay", shard_size=8)
    vec = DummyVecEnv([lambda: ReplayGymWrapper(env, writer)])
    model = build_final_model(vec, n_steps=8, batch_size=4, n_epochs=1, verbose=0)
    metadata = make_checkpoint_metadata(policy_type=FINAL_AGENT, observation_dim=146, action_dim=27)
    cb = FinalCampaign(cfg, env, tmp_path, metadata)
    try:
        model.learn(32, callback=cb)
        assert model.num_timesteps == 32
        assert env.total_interactions > 32  # evaluation + neutral bridges excluded from PPO
        assert base.releases >= 4
        assert (tmp_path / "best.zip").exists()
        assert (tmp_path / "latest.zip").exists()
        assert (tmp_path / "evaluations.csv").exists()
    finally:
        vec.close()
    data = read_session(writer.dir)
    assert data["update_gap"].sum() >= 3
    assert data["evaluation"].sum() > 0
    assert data["match_reward_damage_dealt"].sum() > 0
    from brawl.analysis.plot_final_training import plot_run
    assert plot_run(tmp_path).stat().st_size > 10000


def test_entrypoint_saves_optimizer_metrics_and_resumes_on_cpu(tmp_path, monkeypatch):
    import gymnasium as gym
    from brawl.pipelines.final_agent import run as final_agent
    from brawl.pipelines.final_agent.config import parse_args
    from brawl.pipelines.final_agent.policy import build_final_model
    from brawl.training.checkpoints import make_checkpoint_metadata, FINAL_AGENT

    class Arena(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (146,), dtype=np.float32)
        action_space = gym.spaces.Discrete(27)
        input_controller = SimpleNamespace(reset=lambda: None)
        closed = False

        def reset(self, **kwargs):
            return np.zeros(146, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(146, dtype=np.float32), 0., False, False, event()

        def close(self):
            self.closed = True

    def initialize(source, env, **kwargs):
        kwargs.pop("allow_untyped_skill_checkpoint")
        report = SimpleNamespace(to_dict=lambda: {}, format=lambda: "fake source")
        metadata = make_checkpoint_metadata(policy_type=FINAL_AGENT, observation_dim=146, action_dim=27)
        return build_final_model(env, **kwargs), report, metadata

    monkeypatch.setattr(final_agent, "initialize_final_model", initialize)
    for source in (["--init-from", "unused.zip"], None):
        if source is None:
            source = ["--resume", str(run / "latest.zip")]
        cfg = parse_args(source + [
            "--save-dir", str(tmp_path), "--n-steps", "8", "--batch-size", "4",
            "--timesteps", "8", "--n-epochs", "1", "--eval-every", "0", "--delay", "0"])
        monkeypatch.setattr(final_agent, "parse_args", lambda: cfg)
        arena = Arena()
        monkeypatch.setattr(final_agent, "_make_env", lambda cfg: arena)
        final_agent.main()
        run = max(tmp_path.iterdir(), key=lambda p: p.stat().st_mtime_ns)
        assert (run / "latest.zip").exists()
        assert (run / "ppo" / "progress.csv").exists()
        assert "train/approx_kl" in (run / "ppo" / "progress.csv").read_text()
        assert arena.closed
