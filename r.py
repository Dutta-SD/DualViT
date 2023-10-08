from tp_model import TP_MODEL_MODIFIED_CIFAR10
from vish.utils import test_dl


if __name__ == "__main__":
    net = TP_MODEL_MODIFIED_CIFAR10

    i = iter(test_dl)
    images, fine_labels, broad_labels = next(i)

    print(net)

    net(images)
