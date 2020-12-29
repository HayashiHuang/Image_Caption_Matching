import time
import numpy as np
import torch
import gc
import os
import pdb


class Trainer:
    def __init__(self, data_loader, model, optimizer, args, mode):
        assert mode in ['text', 'mix'], "mode must be one of 'text' and 'mix'."
        self.mode = mode
        self.device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
        self.train_dl = data_loader[0]
        self.valid_dl = data_loader[1]
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
        self.valid_loss = np.zeros(self.max_epoch)
        self.max_batch = {
            'train': len(self.train_dl),
            'valid': len(self.valid_dl)
        }
        self.batch_size = args.batch_size

        # learning policy
        self.lr = args.lr
        self.lr_base = args.lr
        self.warm_up_epoch = args.warm_up_epoch
        self.init_lr = args.init_lr
        self.decay_factor = args.decay_factor
        self.runtime = {
            'iter': 0,
            'batch': 0,
            'train': {
                'text_loss': np.zeros(self.max_batch['train']),
                'text_acc': np.zeros(self.max_batch['train']),
                'img_loss': np.zeros(self.max_batch['train']),
                'img_acc': np.zeros(self.max_batch['train']),
            },
            'valid': {
                'text_loss': np.zeros(self.max_batch['valid']),
                'text_acc': np.zeros(self.max_batch['valid']),
                'img_loss': np.zeros(self.max_batch['valid']),
                'img_acc': np.zeros(self.max_batch['valid']),
            },
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

    def _print_batch_info(self, mode):
        # epoch and batch
        epoch = self.cur_epoch
        epoch_num = self.max_epoch
        batch = self.runtime['batch']
        batch_num = self.max_batch[mode]
        self.runtime['time_cost'] = time.time() - self.runtime['start_time']
        time_per_batch = self.runtime['time_cost'] / batch
        time_left = (batch_num - batch) * time_per_batch

        if self.mode == 'text':
            cur_loss = self.runtime[mode]['text_loss'][batch - 1]
            avg_loss = np.mean(self.runtime[mode]['text_loss'][: batch])
            cur_acc = self.runtime[mode]['text_acc'][batch - 1]
            avg_acc = np.mean(self.runtime[mode]['text_acc'][: batch])
            if mode == 'train':
                info = 'Epoch %d/%d, batch %d/%d: ' % (epoch, epoch_num, batch, batch_num)
                info += 'lr={:.6f}'.format(self.lr) + ', '
            else:
                info = 'Valid %d/%d: ' % (batch, batch_num)
            info += 'cur_loss=%5.6f(%5.6f)' % (cur_loss, avg_loss)
            info += ', accuracy=%.4f(%.4f)' % (cur_acc, avg_acc)
            info += ', %7.3fs/batch, %4.0fs remaining.' % (time_per_batch, time_left)
            print('\r' + info, end='')
        else:
            cur_text_loss = self.runtime[mode]['text_loss'][batch - 1]
            avg_text_loss = np.mean(self.runtime[mode]['text_loss'][: batch])
            cur_text_acc = self.runtime[mode]['text_acc'][batch - 1]
            avg_text_acc = np.mean(self.runtime[mode]['text_acc'][: batch])
            cur_img_loss = self.runtime[mode]['img_loss'][batch - 1]
            avg_img_loss = np.mean(self.runtime[mode]['img_loss'][: batch])
            cur_img_acc = self.runtime[mode]['img_acc'][batch - 1]
            avg_img_acc = np.mean(self.runtime[mode]['img_acc'][: batch])
            if mode == 'train':
                info = 'Epoch %d/%d, batch %d/%d: ' % (epoch, epoch_num, batch, batch_num)
                info += 'lr={:.5f}'.format(self.lr) + ', '
            else:
                info = 'Valid %d/%d: ' % (batch, batch_num)
            info += 'cur_t_loss=%5.4f(%5.4f)' % (cur_text_loss, avg_text_loss)
            info += ', t_acc=%.4f(%.4f)' % (cur_text_acc, avg_text_acc)
            info += 'cur_i_loss=%5.4f(%5.4f)' % (cur_img_loss, avg_img_loss)
            info += ', i_acc=%.4f(%.4f)' % (cur_img_acc, avg_img_acc)
            info += ', %7.3fs/batch, %4.0fs remaining.' % (time_per_batch, time_left)
            print('\r' + info, end='')


    def _adjust_lr(self):
        if self.warm_up_epoch <= 0:
            return
        if self.cur_epoch <= self.warm_up_epoch:
            self.lr += self.lr_step
        else:
            self.lr *= self.decay_factor
            if self.lr < self.lr_base * 0.1:
                self.lr = self.lr_base * 0.1
        optim_state = self.optimizer.state_dict()
        optim_state["param_groups"][0]["lr"] = self.lr
        self.optimizer.load_state_dict(optim_state)

    def _run_one_epoch(self):
        self.runtime['start_time'] = time.time()
        data_loader = self.train_dl
        idx = 0
        for batch in data_loader:
            img, caption = batch
            self.runtime['batch'] = idx + 1
            X_cap = caption.cuda().long()
            X_img = img.cuda().float()
            Y_train = self.model([X_img, X_cap], self.mode)
            if self.mode == 'text':
                batch_loss, accuracy = self.single_model.loss(self.mode, self.cur_epoch)
                self.runtime['train']['text_loss'][idx] = batch_loss.item()
                self.runtime['train']['text_acc'][idx] = accuracy
                self.optimizer.zero_grad()
                batch_loss.backward()
            else:
                img_loss, img_acc, text_loss, text_acc = self.single_model.loss(self.mode, self.cur_epoch)
                self.runtime['train']['text_loss'][idx] = text_loss.item()
                self.runtime['train']['text_acc'][idx] = text_acc
                self.runtime['train']['img_loss'][idx] = img_loss.item()
                self.runtime['train']['img_acc'][idx] = img_acc
                self.optimizer.zero_grad()
                loss = img_loss + text_loss
                loss.backward()


            self.optimizer.step()
            self._print_batch_info('train')
            gc.collect()
            idx += 1
        print('')
        return np.average(self.runtime['train']['text_loss'])

    def _perform_validation(self):
        self.runtime['start_time'] = time.time()
        data_loader = self.valid_dl
        idx = 0
        for batch in data_loader:
            caption, _ = batch
            self.runtime['batch'] = idx + 1
            X_cap = caption.cuda().long()
            pdb.set_trace()
            with torch.no_grad():
                # pdb.set_trace()
                Y_train = self.model([None, X_cap], self.mode)
            batch_loss, accuracy = self.single_model.loss(self.mode, self.cur_epoch)
            self.runtime['valid']['batch_loss'][idx] = batch_loss.item()
            self.runtime['valid']['accuracy'][idx] = accuracy
            self._print_batch_info('valid')
            gc.collect()
            idx += 1
        print('')
        return np.average(self.runtime['valid']['batch_loss'])

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
            #self.model.eval()
            #self.valid_loss[epoch] = self._perform_validation()
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
