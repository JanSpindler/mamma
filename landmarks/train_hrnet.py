# MAMMA env bootstrap — populates os.environ from in-code DEFAULTS and an
# optional .env.local before Hydra's @hydra.main resolves ${oc.env:...}.
# Must precede `import hydra` so the decorator can resolve interpolations.
import sys as _sys
from pathlib import Path as _Path
_mamma_root = _Path(__file__).resolve().parents[1]
if str(_mamma_root) not in _sys.path:
    _sys.path.insert(0, str(_mamma_root))
from inference.env import bootstrap_env as _bootstrap_env
_bootstrap_env()

import os
import os.path as osp


import torch
import hydra

import omegaconf
from omegaconf import OmegaConf, DictConfig
import numpy as np

torch.utils.data._utils.worker.IS_DAEMON = False
os.environ["PYTHONFAULTHANDLER"] = "1"
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar as ProgressBar
from loguru import logger
from configs import constants as _C
from utils.util import init_random_seed, set_random_seed
from lib.models import build_model
try:
    torch.set_float32_matmul_precision('high')
except AttributeError:
    pass
torch.multiprocessing.set_sharing_strategy('file_system')
CUR_PATH = osp.dirname(__file__)


@hydra.main(version_base=None, config_path="configs/train/models_2d", config_name="config.yaml")
def main(cfg: DictConfig):

    OmegaConf.register_new_resolver("mult", lambda x,y: x*y)
    OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
    OmegaConf.register_new_resolver("div", lambda x, y: x // y)
    OmegaConf.register_new_resolver("concat", lambda x: np.concatenate(x))
    OmegaConf.register_new_resolver("sorted", lambda x: np.argsort(x))

    task = cfg.task
    exp_name = cfg.exp_name
    work_dir = osp.join(CUR_PATH, cfg.work_dir, 'train', task, exp_name)
    log_dir = osp.join(work_dir, 'logs')
    viz_dir = osp.join(work_dir, 'viz')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    logger.add(
        os.path.join(log_dir, 'train.log'),
        level='INFO',
        colorize=False,
    )

    # set cudnn_benchmark
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    seed = init_random_seed(cfg.seed)
    set_random_seed(seed, deterministic=cfg.deterministic)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(torch.cuda.get_device_properties(device))

    experiment_loggers = []
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        log_graph=False,
    )
    experiment_loggers.append(tb_logger)

    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    tb_logger.log_hyperparams(cfg_dict)

    # wandb is optional: enable only when credentials are present, so the code
    # still runs end-to-end for anyone without a wandb account.
    if os.environ.get("WANDB_API_KEY"):
        try:
            wandb_logger = WandbLogger(project=cfg.task, name=exp_name)
            wandb_logger.log_hyperparams(cfg_dict)
            experiment_loggers.append(wandb_logger)
        except Exception as e:
            logger.warning(f"wandb logger failed to initialize ({e}); continuing with TensorBoard only.")
    else:
        logger.info("WANDB_API_KEY not set; skipping wandb logger. Metrics go to TensorBoard.")

    ckpt_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{exp_name}",
        monitor='train/loss',
        verbose=True,
        save_top_k=3,
        mode='min',
        every_n_train_steps=1000,
        save_last=True,
    )

    model = build_model(cfg, viz_dir=viz_dir).to(device)
    logger.info(f'Loaded pretrained backbone weights from {_C.PATHS.PRETRAINED_VITPOSE_CKPT_PTH}')

    if cfg.model_checkpoint is not None:
        logger.info(f'Loading model checkpoint from {cfg.model_checkpoint}')
        model.load_state_dict(torch.load(cfg.model_checkpoint, weights_only=False)['state_dict'], strict=False)
    else:
        logger.info(f'no model checkpoint provided')
    try: # Version 1.6
        trainer = pl.Trainer(
            gpus=cfg.gpus_n,
            strategy= cfg.strategy,
            max_steps=cfg.max_steps,
            logger=experiment_loggers,
            callbacks=[ckpt_callback, ProgressBar(refresh_rate=1)],
            default_root_dir=work_dir,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=1,
            on_gpu=True,
        )
    except Exception as e:
        logger.warning(f"Lightning 1.x trainer init failed ({e}); falling back to alternate config")
        trainer = pl.Trainer(
            overfit_batches=1 if cfg.debug_mode else 0,
            devices=int(cfg.gpus_n),
            max_steps=cfg.max_steps,
            strategy=cfg.strategy,
            logger=experiment_loggers,
            callbacks=[ckpt_callback, ProgressBar(refresh_rate=1)],
            default_root_dir=work_dir,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=1,
            accumulate_grad_batches=cfg.acc_grads,
            profiler='simple',
        )

    logger.info('*** Started training ***')

    trainer.fit(model)


if __name__ == '__main__':
    main()
