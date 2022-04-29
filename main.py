from Data import *
from Optim import *
import torch.optim as optim
from GRU import *


def get_model(model, model_params):
    models = {
        "rnn": RNN,
        "lstm": LSTM,
        "gru": GRU,
    }
    return models.get(model.lower())(**model_params)




def run_program():
    obj1 = DataClean("C:\\Users\\KLEMS\\Downloads\\fin1.csv")
    df = obj1.get_data()
    df_new = obj1.manipulate(df)
    targets = obj1.get_targets(df_new)

    train = obj1.get_train(df_new)
    train = train.drop('Unnamed: 0', axis=1)

    xtr, xte, ytr, yte = obj1.train_test(train, targets)

    yte = yte.to_frame()
    ytr = ytr.to_frame()
    xtr, ytr = obj1.to_numpy(xtr, ytr)
    xte, yte = obj1.to_numpy(xte, yte)
    train_loader, test_loader, test_loader_one = obj1.data_to_tensors(xtr, xte, ytr, yte, 3)

    return train_loader, test_loader, test_loader_one, len(train.columns)
    # return 1, 2, 3


if __name__ == '__main__':
    train_loader, test_loader, test_loader_one, x = run_program()
    output_dim = 1
    model_path = 'model.pth'
    check = 0
    layer = 4
    num_epochs = 50
    learning_rate = 0.001
    input_dim = x
    hidden_dim = 4
    drop_put = 0.2

    model_params = {'input_size': input_dim,
                    'hidden_size': hidden_dim,
                    'num_layers': layer,
                    'output_dem': output_dim,
                    'prob_drop': drop_put}

    if check == 0:
        model = get_model('rnn', model_params)

    else:
        model = torch.load(model_path)

    lossFunc = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    opt = Optim(model=model, loss_func=lossFunc, optim=optimizer)
    print("The input dimension: " + str(x))

    opt.train(train_loader, test_loader, batch_size=3, num_epochs=num_epochs, features=input_dim)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
