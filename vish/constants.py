# Constants
WEIGHT_FOLDER_PATH = "./checkpoints"
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 200
BROAD_EPOCHS = 2
FINE_EPOCHS = 2
IMG_SIZE = 224
MIN_LR = 1e-8
DESC = "vit-decomposed-cifar10"
ALT_FREQ = 50
VIT_PRETRAINED_MODEL_1 = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
VIT_PRETRAINED_MODEL_2 = "google/vit-base-patch16-224"
CIFAR10_NUM_BROAD = 2
CIFAR10_NUM_FINE = 10
IS_TEST_RUN = False
LOAD_CKPT = False
DATE_TIME_FORMAT = "%Y_%m_%d_%H_%M_%S"
MOMENTUM = 0.9
CIFAR100_FINE_2_BROAD_MAP = {
    0: 4,
    1: 1,
    2: 14,
    3: 8,
    4: 0,
    5: 6,
    6: 7,
    7: 7,
    8: 18,
    9: 3,
    10: 3,
    11: 14,
    12: 9,
    13: 18,
    14: 7,
    15: 11,
    16: 3,
    17: 9,
    18: 7,
    19: 11,
    20: 6,
    21: 11,
    22: 5,
    23: 10,
    24: 7,
    25: 6,
    26: 13,
    27: 15,
    28: 3,
    29: 15,
    30: 0,
    31: 11,
    32: 1,
    33: 10,
    34: 12,
    35: 14,
    36: 16,
    37: 9,
    38: 11,
    39: 5,
    40: 5,
    41: 19,
    42: 8,
    43: 8,
    44: 15,
    45: 13,
    46: 14,
    47: 17,
    48: 18,
    49: 10,
    50: 16,
    51: 4,
    52: 17,
    53: 4,
    54: 2,
    55: 0,
    56: 17,
    57: 4,
    58: 18,
    59: 17,
    60: 10,
    61: 3,
    62: 2,
    63: 12,
    64: 12,
    65: 16,
    66: 12,
    67: 1,
    68: 9,
    69: 19,
    70: 2,
    71: 10,
    72: 0,
    73: 1,
    74: 16,
    75: 12,
    76: 9,
    77: 13,
    78: 15,
    79: 13,
    80: 16,
    81: 19,
    82: 2,
    83: 4,
    84: 6,
    85: 19,
    86: 5,
    87: 5,
    88: 8,
    89: 19,
    90: 18,
    91: 1,
    92: 2,
    93: 15,
    94: 6,
    95: 0,
    96: 17,
    97: 8,
    98: 14,
    99: 13,
}
