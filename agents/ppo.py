"""PPO trainer (Section 1.7 of the methodology).

We wrap stable-baselines3's ``PPO`` because it is the canonical, peer-reviewed
implementation of Schulman et al. (2017) — its clipped surrogate objective is
exactly Eq. 1.7 in the methodology, with ``clip_range`` corresponding to ε.

Training environment is the ``CryptoTradingEnv`` with the LSTM signal injected
into the state space (the hybrid integration from Section 1.8).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from utils.logging import get_logger

log = get_logger(__name__)


def make_env_fn(
    raw_df,
    scaler,
    lstm_signals,
    cfg,
    seed: int = 0,
) -> Callable[[], "gym.Env"]:
    """Return a thunk that builds a fresh env, used by SB3 vectorised envs."""
    from env.trading_env import CryptoTradingEnv  # local import to avoid hard gym dep at import time

    def _thunk():
        env = CryptoTradingEnv(raw_df=raw_df, scaler=scaler, lstm_signals=lstm_signals, cfg=cfg, seed=seed)
        return env
    return _thunk


def train_ppo(
    cfg,
    train_raw,
    val_raw,
    scaler,
    train_lstm_signals: np.ndarray,
    val_lstm_signals: np.ndarray,
    save_path: str | Path,
    vecnorm_path: str | Path,
):
    """Train PPO and persist the policy + observation-normalisation stats."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    save_path = Path(save_path)
    vecnorm_path = Path(vecnorm_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def make_train():
        env = make_env_fn(train_raw, scaler, train_lstm_signals, cfg, seed=cfg.seed)()
        return Monitor(env)

    def make_eval():
        env = make_env_fn(val_raw, scaler, val_lstm_signals, cfg, seed=cfg.seed + 1)()
        return Monitor(env)

    train_env = DummyVecEnv([make_train])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=cfg.ppo.gamma)

    eval_env = DummyVecEnv([make_eval])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=cfg.ppo.gamma, training=False)
    # Share running statistics with the training env
    eval_env.obs_rms = train_env.obs_rms

    policy_kwargs = dict(net_arch=list(cfg.ppo.policy_kwargs.net_arch))

    model = PPO(
        policy=cfg.ppo.policy,
        env=train_env,
        learning_rate=cfg.ppo.learning_rate,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_range=cfg.ppo.clip_range,
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=cfg.seed,
        device=cfg.device if cfg.device != "auto" else "auto",
    )

    stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=3, verbose=1)
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(save_path.parent / "ppo_best"),
        log_path=str(save_path.parent / "ppo_logs"),
        eval_freq=cfg.ppo.eval_freq,
        n_eval_episodes=cfg.ppo.n_eval_episodes,
        deterministic=True,
        render=False,
        callback_after_eval=stop_cb,
    )

    log.info("Starting PPO training for %d timesteps", cfg.ppo.total_timesteps)
    model.learn(total_timesteps=int(cfg.ppo.total_timesteps), callback=eval_cb, progress_bar=False)

    model.save(str(save_path))
    train_env.save(str(vecnorm_path))
    log.info("PPO saved to %s; VecNormalize stats to %s", save_path, vecnorm_path)
    return model


def load_ppo(model_path: str | Path, vecnorm_path: str | Path, raw_df, scaler, lstm_signals, cfg):
    """Reconstruct the PPO model + observation normaliser for inference."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def thunk():
        from env.trading_env import CryptoTradingEnv
        return Monitor(CryptoTradingEnv(raw_df, scaler, lstm_signals, cfg))

    env = DummyVecEnv([thunk])
    env = VecNormalize.load(str(vecnorm_path), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env, device=cfg.device if cfg.device != "auto" else "auto")
    return model, env
