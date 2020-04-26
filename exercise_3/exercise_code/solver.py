import numpy as np
import torch


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        #######################################################################
        # TODO:                                                               #
        # Write your own personal training method for our solver. In each     #
        # epoch iter_per_epoch shuffled training batches are processed. The   #
        # loss for each batch is stored in self.train_loss_history. Every     #
        # log_nth iteration the loss is logged. After one epoch the training  #
        # accuracy of the last mini batch is logged and stored in             #
        # self.train_acc_history. We validate at the end of each epoch, log   #
        # the result and store the accuracy of the entire validation set in   #
        # self.val_acc_history.                                               #
        #                                                                     #
        # Your logging could like something like:                             #
        #   ...                                                               #
        #   [Iteration 700/4800] TRAIN loss: 1.452                            #
        #   [Iteration 800/4800] TRAIN loss: 1.409                            #
        #   [Iteration 900/4800] TRAIN loss: 1.374                            #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                           #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                           #
        #   ...                                                               #
        #######################################################################

        for epoch in range(num_epochs):
            iter_count = 0
            for data, targets in train_loader:
                optim.zero_grad()
                out = model(data)

                 #print(out.shape, targets.shape)
                 #out_new = out.transpose(1,2)
                 #out_new = out_new.transpose(2,3)
                 #print(out_new.shape)
                 #out_new = torch.Tensor(list(map(lambda x: (list(map(lambda y: (list(map(lambda z: torch.max(z, 0)[1],y))),x))) ,out_new)))
                #print(out.shape)
                #N, c, h, w = out.shape
                #out_new = out.transpose(1,2)
                #out_new = out_new.transpose(2,3)
                #out_new = out_new.reshape(-1, c).argmax(axis=1).reshape(N, h, w)
                #out_new = out_new.float()
                #print(out_new.shape, targets.shape)
                 #print(out_new.shape)
                 #print(out.shape)
                loss = self.loss_func(out, targets)
                loss.backward()
                optim.step()

                if log_nth > 0:
                    if iter_count % log_nth == 0:
                        print('[Iteration', iter_count, '] Train loss: ',loss)
                self.train_loss_history.append(loss)
                 #print((out_new==targets).shape)

                _, preds = torch.max(out, 1)
                targets_mask = targets >= 0
                batch_acc = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
                #batch_acc = np.mean(np.array(out_new == targets))
                iter_count += 1


            self.train_acc_history.append(batch_acc)
            val_scores = np.array([])
            for data, targets in val_loader:
                outputs = model.forward(data)
                val_loss = self.loss_func(outputs, targets)
                _, preds = torch.max(outputs, 1)
                targets_mask = targets >= 0
                val_scores = np.append(val_scores, np.mean((preds == targets)[targets_mask].data.cpu().numpy()))

            val_acc = np.mean(val_scores)
            self.val_acc_history.append(val_acc)
            print('Epoch ', epoch, 'vall acc/loss: ', val_acc, val_loss )
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        print('FINISH.')
