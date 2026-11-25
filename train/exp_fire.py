"""YOLOX-Tiny для огня/дыма на D-Fire. Запуск из папки YOLOX:

python tools/train.py -f <путь>/exp_fire.py -d 1 -b 16 --fp16 -c <путь>/yolox_tiny.pth
Перед запуском: set DFIRE=<папка dfire_yolo>, где лежат annotations/ (dfire_to_coco.py).
"""
import os

from yolox.exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.depth, self.width = 0.33, 0.375          # tiny
        self.num_classes = 2                          # 0 fire, 1 smoke
        self.data_dir = os.environ.get("DFIRE", "dfire_yolo")
        self.train_ann, self.val_ann, self.test_ann = "train.json", "val.json", "test.json"
        self.input_size = self.test_size = (416, 416)
        self.multiscale_range = 3
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        self.max_epoch = int(os.environ.get("EPOCHS", 30))
        self.no_aug_epochs = 5
        self.warmup_epochs = 2
        self.eval_interval = 5
        self.print_interval = 50
        self.data_num_workers = 4
        self.exp_name = "fire_tiny"
        self.output_dir = os.environ.get("OUT", "./runs")

    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(data_dir=self.data_dir, json_file=self.train_ann, name="train/images",
                           img_size=self.input_size,
                           preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
                           cache=cache, cache_type=cache_type)

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        test = kwargs.get("testdev", False)
        return COCODataset(data_dir=self.data_dir, json_file=self.test_ann if test else self.val_ann,
                           name="test/images" if test else "val/images",
                           img_size=self.test_size, preproc=ValTransform(legacy=kwargs.get("legacy", False)))
