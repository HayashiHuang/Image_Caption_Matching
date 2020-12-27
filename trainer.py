import time
import numpy as np
import torch
import gc
import os


class Trainer:
    def __init__(self, data_loader, model, optimizer, args, mode):
        assert mode in ['text', 'mix'], "mode must be one of 'text' and 'mix'."
        self.mode = mode
        self.device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        # if torch.cuda.device_count() > 1:
        #     self.single_model = self.model.module
        # else:
        self.single_model = self.model
        # work path:
        self.work_dir = args.work_dir + '/work_dir'
        os.makedirs(self.work_dir, exist_ok=True)

        # epoch and batch:
        if self.mode == 'text':
            self.max_epoch = args.text_epoch
        else:
            self.max_epoch = args.epoch
        self.cur_epoch = 0
        self.epoch_loss = np.zeros(self.max_epoch)
        self.max_batch = len(data_loader)
        self.batch_size = args.batch_size

        # learning policy
        self.lr = args.lr
        self.warm_up_epoch = args.warm_up_epoch
        self.init_lr = args.init_lr
        self.decay_factor = args.decay_factor
        self.runtime = {
            'batch': 0,
            'batch_loss': np.zeros(self.max_batch),
            'accuracy': 0,
            "start_time": 0.1,
            "time_cost": 0.1
        }
        # reset learning rate to init_lr if warm-up is set
        if self.warm_up_epoch > 0:
            self.lr_step = (self.lr - self.init_lr) / self.warm_up_epoch
            self.lr = self.init_lr
            optim_state = self.optimizer.state_dict()
            optim_state["param_groups"][0]["lr"] = self.lr
            self.optimizer.load_state_dict(optim_state)

    def _print_batch_info(self):
        # epoch and batch
        epoch = self.cur_epoch
        epoch_num = self.max_epoch
        batch = self.runtime['batch']
        batch_num = self.max_batch
        # loss
        cur_loss = self.runtime['batch_loss'][batch - 1]
        avg_loss = np.mean(self.runtime['batch_loss'][: batch])
        # time
        self.runtime['time_cost'] = time.time() - self.runtime['start_time']
        time_per_batch = self.runtime['time_cost'] / batch
        time_left = (batch_num - batch) * time_per_batch

        info = 'Epoch %d/%d, batch %d/%d: ' % (epoch, epoch_num, batch, batch_num)
        info += 'lr={:.6f}'.format(self.lr)
        info += ', cur_loss=%5.6f, avg_loss=%5.6f' % (cur_loss, avg_loss)
        info +=', accuracy={:.4f}'.format(self.runtime['accuracy'])
        info += '%7.3fs/batch, %4.0fs remaining.' % (time_per_batch, time_left)
        

        print('\r' + info, end='')

    def _adjust_lr(self):
        if self.warm_up_epoch <= 0:
            return
        if self.cur_epoch <= self.warm_up_epoch:
            self.lr += self.lr_step
        else:
            self.lr *= self.decay_factor
        optim_state = self.optimizer.state_dict()
        optim_state["param_groups"][0]["lr"] = self.lr
        self.optimizer.load_state_dict(optim_state)

    def _run_one_epoch(self):
        self.runtime['start_time'] = time.time()
        data_loader = self.data_loader
        idx = 0
        for batch in data_loader:
            img, caption = batch
            self.runtime['batch'] = idx + 1
            X_cap = caption.cuda().long()
            X_img = img.cuda().float()
            Y_train = self.model([X_img, X_cap], self.mode)
            batch_loss, accuracy = self.single_model.loss(self.mode, self.cur_epoch)
            self.runtime['batch_loss'][idx] = batch_loss.item()
            self.runtime['accuracy'] = accuracy
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            self._print_batch_info()
            gc.collect()
            idx += 1
        print('')
        return np.average(self.runtime['batch_loss'])

    def _save_model(self):
        model = self.single_model.state_dict()
        latest_path = self.work_dir + '/' + self.mode + '.pth'
        torch.save(model, latest_path)

    def _save_loss(self):
        train_loss_path = self.work_dir + '/' + self.mode + '_train_loss.txt'
        np.savetxt(train_loss_path, self.epoch_loss, fmt='%.8f')

    def train(self):
        print('****', self.mode)
        print('Total epoch: %d' % self.max_epoch)
        print('Batch size: %d' % self.batch_size)
        print('Learning rate: %f' % self.lr)
        print('\nstart training...')
        start_time = time.time()
        for epoch in range(self.max_epoch):
            self.cur_epoch = epoch + 1
            self.model.train()
            self.epoch_loss[epoch] = self._run_one_epoch()
            self._adjust_lr()
        self._save_model()
        self._save_loss()
        time_cost = time.time() - start_time
        hour = time_cost // 3600
        minute = (time_cost % 3600) // 60
        second = (time_cost % 3600) % 60
        print('time cost: %.0fh%.0fm%.0fs' % (hour, minute, second))
        print('Training finished!')

        return self.model
