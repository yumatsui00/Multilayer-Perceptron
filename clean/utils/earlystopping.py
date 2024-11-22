class EarlyStop():
    """earlystop class"""
    def __init__(self, loss_border=0.1, patience=5):
        """params:
            loss border : (float)
                border which loss needs to be lower than at least
            patience : (int)
                the count to meet the condition for earlystopping
        """
        self.loss_border = loss_border
        self.count = 0
        self.patience = patience
        self.best_weight = []
        self.prev_loss = float('inf')
        self.min_val_loss = float('inf')
        self.prev_val_loss = float('inf')

    def evaluate(self, current_loss, current_val_loss, current_weight):
        """evaluate if model needs to earlystop
        params :
            current_loss : (float)
                current loss from train data
            current_val_loss ": (float)
                current loss from test data
            current-weight : (array)
                current weight for saving weight to recover
        """
        self.min_val_loss = current_val_loss if current_val_loss < self.min_val_loss else self.min_val_loss
        if current_loss > self.loss_border:
            self.reset(current_loss, current_val_loss)
            return False
        if abs(self.prev_loss - current_loss) > 0.01:
            self.reset(current_loss, current_val_loss)
            return False
        if current_val_loss > self.prev_val_loss:
            if self.count == 0:
                self.save_weight(current_weight)
            self.count += 1
            self.prev_loss = current_loss
            self.prev_val_loss = current_val_loss
        else:
            self.reset(current_loss, current_val_loss)
        if self.count >= self.patience:
            return True
        return False


    def reset(self, loss, val_loss):
        """reset parameter while earlystopping"""
        self.count = 0
        self.prev_loss = loss
        self.prev_val_loss = val_loss


    def save_weight(self, weight):
        """save weight to self
        params
            weight : (array)
                weight to save
        """
        self.best_weight = weight


    def recovery(self):
        """return the best weight when the earystop is activated"""
        return self.best_weight
