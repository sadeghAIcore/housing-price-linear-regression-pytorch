
import torch
import pandas as pd 
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('../data/california_housing.csv')

device = "cuda" if torch.cuda.is_available() else "cpu"

X = data.drop('price' , axis=1).values
y = data['price'].values.reshape(-1,1)

model = LinearRegression(len(X[0])).to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


scaler_instance = StandardScaler()
X_train = scaler_instance.fit_transform(X_train) 
X_test = scaler_instance.transform(X_test)  


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters() , lr=0.003)

X_train = torch.from_numpy(X_train).type(torch.float)
X_test = torch.from_numpy(X_test).type(torch.float)
y_train = torch.from_numpy(y_train).type(torch.float)
y_test = torch.from_numpy(y_test).type(torch.float)

# building the train function

losses = []
epochs = 200

X_train , X_test = X_train.to(device) , X_test.to(device)
y_train , y_test = y_train.to(device) , y_test.to(device)



for epoch in range(epochs +  1) :
    model.train()

    y_preds = model(X_train)

    loss = loss_fn(y_preds , y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if epoch % 10 == 0 :
        print(f'epoch {epoch+1} of {epochs} | loss : {loss}')

    losses.append(loss)
# saving the models into the models folder
torch.save(model.state_dict(),"../models/model.pth")
print("✅ Model Saved")


# plot and saving the plot
numpy_losses = np.array([l.cpu().detach().item() if hasattr(l, 'cpu') else l for l in losses])
plt.plot(numpy_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_curve.png")
plt.show()
