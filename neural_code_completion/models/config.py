class SmallConfig(object):
    """Small config.  get best result as 0.733 """
    init_scale = 0.05
    learning_rate = 0.001
    max_grad_norm = 5
    num_layers = 1
    num_steps = 50
    attn_size = 50
    hidden_sizeN = 300
    hidden_sizeT = 500
    sizeH = 800
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.6
    batch_size = 64


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.05
    learning_rate = 0.001
    max_grad_norm = 5
    num_layers = 1
    num_steps = 50
    attn_size = 50
    hidden_sizeN = 300
    hidden_sizeT = 500
    sizeH = 800
    max_epoch = 1
    max_max_epoch = 2
    keep_prob = 1.0
    lr_decay = 0.6
    batch_size = 1


class BestConfig(object):
    """Best Config according to the paper."""
    init_scale = 0.05
    learning_rate = 0.001
    max_grad_norm = 5
    num_layers = 1
    num_steps = 50
    attn_size = 50
    hidden_sizeN = 300
    hidden_sizeT = 1200
    sizeH = 1500
    max_epoch = 1
    max_max_epoch = 8
    keep_prob = 1.0
    lr_decay = 0.6
    batch_size = 128

class ExperimentalConfig(object):
    """Intermediate config used for experiments."""
    init_scale = 0.05
    learning_rate = 0.001
    max_grad_norm = 5
    num_layers = 1
    num_steps = 50
    attn_size = 50
    hidden_sizeN = 300
    hidden_sizeT = 1200
    sizeH = 1500
    max_epoch = 1
    max_max_epoch = 8
    keep_prob = 0.6
    lr_decay = 0.6
    batch_size = 128