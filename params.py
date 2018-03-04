class TrainingParams:
    def __init__(self):
        self.style_path = 'data/style.jpg'
        self.mask_path = 'data/style_mask.jpg'
        self.input_shape = [256,256,3]
        self.num_colors = 4