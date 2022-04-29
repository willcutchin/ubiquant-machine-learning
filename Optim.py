from datetime import datetime
from RNN import *
from LSTM import *
import numpy as np
from GRU import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Optim:
    def __init__(self, model, loss_func, optim):
        self.model = model
        self.loss_func = loss_func
        self.optim = optim
        self.train_losses = []
        self.val_losses = []
        # self.FILE = "model.pth"

    def training_step(self, x, y):

        self.model.train()
        y_pred = self.model(x)
        # print("Y Predict is: " + str(y_pred))
        # print(len(x))
        # print(len(y))
        # print(len(y_pred))
        self.optim.zero_grad()
        loss = self.loss_func(y_pred, y)
        loss.backward()
        self.optim.step()


        return loss.item()

    def train(self, trainer, valid, batch_size, num_epochs, features):
        model_path = 'model.pth'
        for epoch in range(1, num_epochs+1):
            batch_loss = []
            i = 0
            for x, y in trainer:
                i += 1
                if x.shape[0] == 3:
                    x = x.view([batch_size, -1, features]).to(device)
                    y = y.to(device)
                    loss = self.training_step(x, y)
                    batch_loss.append(loss)
                    # if i%100 == 0:

                        # print(i)



            # print(len(batch_loss))
            training_loss = np.mean(batch_loss)
            # print("The training loss is: " + str(training_loss))
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_loss = []
                j = 0
                for xVal, yVal in valid:
                    # print(xVal.shape)
                    # j +=1
                    if xVal.shape[0] == 3:
                        xVal = xVal.view([batch_size, -1, features]).to(device)
                        yVal = yVal.to(device)
                        ypred = self.model(xVal)
                        valLoss = self.loss_func(yVal, ypred)
                        batch_val_loss.append(valLoss)

                valid_loss = np.mean(batch_val_loss)
                self.val_losses.append(valid_loss)

            if epoch % 1 == 0:
                print(f"[{epoch}/{num_epochs}] Training loss: {training_loss}\t Valid loss: {valid_loss}")

        torch.save(self.model, model_path)

    def eval(self, test_loader, batch_size, features):
        with torch.no_grad():
            pred = []
            values = []
            for x, y in test_loader:
                x = x.view([batch_size, -1, features]).to(device)
                y = y.to(device)
                self.model.eval()
                ypred = self.model(x)
                pred.append(ypred.to(device).detach().numpy())
                values.append(y.to(device).detach().numpy())

        return pred, values
