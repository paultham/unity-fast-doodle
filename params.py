class TrainingParams:
    def __init__(self):
        self.style_weight = 1.0
        self.style_path = 'data/style.jpg'
        self.mask_path = 'data/style_mask.jpg'
        self.input_shape = [256,256,3]
        self.num_colors = 4
        self.save_path = 'summaries'
        self.summary_step = 2
        self.log_step = 1
        self.learn_rate = 0.001

        self.train_path = 'data/mask.trf'
        self.total_train_sample = 10
        self.batch_size = 1
        self.num_epoch=1
        self.read_thread = 1