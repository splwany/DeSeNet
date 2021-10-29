import sys
from pathlib import Path

import wandb

FILE = Path(__file__).resolve()
ROOT = FILE.parents[4]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to path

from scripts.train import train, parse_opt
from core.utils.general import increment_path
from core.utils.torch_utils import select_device
from core.utils.callbacks import Callbacks


def sweep():
    wandb.init()
    # Get hyp dict from sweep agent
    hyp_dict = vars(wandb.config).get("_items")
    assert hyp_dict is not None

    # Workaround: get necessary opt args
    opt = parse_opt(known=True)
    opt.batch_size = hyp_dict.get("batch_size")
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok or opt.evolve))
    opt.epochs = hyp_dict.get("epochs")
    opt.nosave = True
    opt.data = hyp_dict.get("data")
    device = select_device(opt.device, batch_size=opt.batch_size)

    # train
    train(hyp_dict, opt, device, callbacks=Callbacks())


if __name__ == "__main__":
    sweep()
