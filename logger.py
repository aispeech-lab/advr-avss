from tensorboardX import SummaryWriter


class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def log_train(self, loss_ss, iteration):
        self.add_scalar("train.loss_ss", loss_ss, iteration)

    def log_train_sisnr(self, loss_ss, iteration):
        self.add_scalar("train.loss_ss", loss_ss, iteration)

    def log_train_sisnr_msnr(self, loss_sisnr, loss_mse, loss_total, iteration):
        self.add_scalar("train.loss_sisnr", loss_sisnr, iteration)
        self.add_scalar("train.loss_msnr", loss_mse, iteration)
        self.add_scalar("train.loss_total", loss_total, iteration)

    def log_test(self, loss_ss, iteration):
        self.add_scalar("test.loss_ss", loss_ss, iteration)

    def log_test_sisnr(self, loss_ss, iteration):
        self.add_scalar("test.loss_ss", loss_ss, iteration)

    def log_test_sisnr_msnr(self, loss_sisnr, loss_mse, loss_total, iteration):
        self.add_scalar("test.loss_sisnr", loss_sisnr, iteration)
        self.add_scalar("test.loss_msnr", loss_mse, iteration)
        self.add_scalar("test.loss_total", loss_total, iteration)

    def log_val(self, loss_ss, iteration):
        self.add_scalar("val.loss_ss", loss_ss, iteration)
    
    def log_lr(self, lr, iteration):
        self.add_scalar("model.lr", lr, iteration)