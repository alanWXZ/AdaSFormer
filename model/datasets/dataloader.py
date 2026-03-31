from config import config
from utils.img_utils import normalize


class TrainPre(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img, hha):
        img = normalize(img, self.img_mean, self.img_std)
        hha = normalize(hha, self.img_mean, self.img_std)

        p_img = img.transpose(2, 0, 1)
        p_hha = hha.transpose(2, 0, 1)

        extra_dict = {'hha_img': p_hha}

        return p_img, extra_dict


class ValPre(object):
    def __call__(self, img, hha):
        extra_dict = {'hha_img': hha}
        return img, extra_dict


def get_train_loader(engine, dataset, s3client=None):
    data_setting = {'root': config.root,
                    'gt_root': config.gt_root_folder,
                    'hha_root': config.hha_root_folder,
                    'mapping_root': config.mapping_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}

    train_preprocess = TrainPre(config.image_mean, config.image_std)

    train_dataset = dataset(
        data_setting, "train", train_preprocess, file_length=None, s3client=s3client
    )

    return train_dataset
