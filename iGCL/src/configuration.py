
class Config:
    def __init__(self):
        self.model = 'IDGCL'
        self.device = 0
        self.lr = 0.001
        self.dropout = 0.0
        self.eval_freq = 5

        self.root = r'./Data'
        self.dataset = 'WikiCS'
        self.hidden_layers = [1024]
        self.pred_hid = 2048
        self.topk = 6
        self.lambd = 5e-3
        self.epochs = 1000
        self.mad = 0.9

