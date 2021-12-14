import pickle
import numpy as np
from PIL import Image
import torchvision

from torch.utils.data.dataset import Dataset

class Cifar(Dataset):
    def __init__(self, path='data/cifar-10-batches-py/', transform=None, train=True, samples=None, balance=True):

        self.transform = transform
        self.cls_num_list = []
        if train:
            train_idx = [1, 2, 3, 4, 5]

            # training data
            training_data = []
            training_label = []
            for idx in train_idx:
                data_path = path + 'data_batch_' + str(idx)
                with open(data_path, 'rb') as fp:
                    dict = pickle.load(fp, encoding='bytes')
                    labels = dict[b'labels']
                    data = dict[b'data'].reshape(-1, 3, 32, 32)
                    training_data.append(data)
                    training_label.append(labels)
            self.data = np.concatenate(training_data, axis=0)
            self.data = self.data.transpose((0, 2, 3, 1))
            self.label = np.concatenate(training_label, axis=0)

            if samples is not None:
                class_labels = list(range(10))
                if balance:
                    weights = [0.1] * 10
                else:
                    #weights = [0.1, 0.3, 0.3, 0.1, 0.1, 0.05, 0.03, 0.01, 0.008, 0.002]
                    weights = [0.4, 0.24, 0.14, 0.08, 0.05, 0.04, 0.03, 0.01, 0.006, 0.004]
                data_ = []
                label_ = []
                for l in class_labels:
                    label_mask = (self.label == l)
                    masked_images = self.data[label_mask, :, :, :]
                    masked_labels = self.label[label_mask]
                    num_samples_per_class = int(samples * weights[l])
                    masked_images = masked_images[:num_samples_per_class, :, :, :]
                    masked_labels = masked_labels[:num_samples_per_class]
                    data_.append(masked_images)
                    label_.append(masked_labels)
                    self.cls_num_list.append(masked_images.shape[0])
                self.data = np.concatenate(data_, axis=0)
                self.label = np.concatenate(label_, axis=0)

        else:
            with open(path + 'test_batch', 'rb') as fp:
                dict = pickle.load(fp, encoding='bytes')
                labels = dict[b'labels']
                data = dict[b'data'].reshape(-1, 3, 32, 32)
                self.data = data.transpose((0, 2, 3, 1))
                self.label = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.label[index]
        return (img, label)

    def get_img_num_per_class(self):
        return self.cls_num_list


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



if __name__ == '__main__':
    x = Cifar()
    data = x.get_batched_train()
