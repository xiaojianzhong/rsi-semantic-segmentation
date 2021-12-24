import numpy as np


class Metric:
    def __init__(self, num_classes, ignore_indexes=[]):
        self.num_classes = num_classes
        self.ignore_indexes = ignore_indexes
        self.matrix = np.zeros((self.num_classes, self.num_classes))

    def reset(self):
        self.matrix.fill(0)

    def add(self, pred, label):
        mask = (label >= 0) & (label < self.num_classes)
        for ignore_index in self.ignore_indexes:
            mask &= (label != ignore_index)

        count = np.bincount(self.num_classes * label[mask] + pred[mask], minlength=self.num_classes ** 2)
        self.matrix += count.reshape((self.num_classes, self.num_classes))

    def PA(self):
        acc = np.diag(self.matrix).sum() / self.matrix.sum()
        return acc

    def PAs(self):
        accs = np.diag(self.matrix) / self.matrix.sum(axis=1)
        return accs

    def mPA(self):
        accs = self.PAs()
        acc = accs.mean()
        return acc

    def IoUs(self):
        intersection = np.diag(self.matrix)
        union = np.sum(self.matrix, axis=0) + np.sum(self.matrix, axis=1) - np.diag(self.matrix)
        iou = intersection / union
        return iou

    def mIoU(self):
        ious = self.IoUs()
        iou = ious.mean()
        return iou
