class TrainingParams:
    def __init__(self):
        self.style_weight = 1.0
        self.style_path = 'data/style.jpg'
        self.mask_path = 'data/style_mask.jpg'
        self.input_shape = [256,256,3]
        self.num_colors = 4
        self.save_path = 'summaries'
        self.summary_step = 100
        self.log_step = 100
        self.learn_rate = 0.001
