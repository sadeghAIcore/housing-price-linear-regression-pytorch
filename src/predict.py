import torch

from model import LinearRegression

data = pd.read_csv('../data/california_housing.csv')


X = data.drop('price' , axis=1).values
y = data['price'].values.reshape(-1,1)

model = LinearRegression(len(X[0])).to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


scaler_instance = StandardScaler()
X_train = scaler_instance.fit_transform(X_train) 
X_test = scaler_instance.transform(X_test)  


X_train = torch.from_numpy(X_train).type(torch.float)
X_test = torch.from_numpy(X_test).type(torch.float)
y_train = torch.from_numpy(y_train).type(torch.float)
y_test = torch.from_numpy(y_test).type(torch.float)

loss_fn = nn.MSELoss()

# building the predict function


model = LinearRegression(len(X[0]))
model.load_state_dict(torch.load('../models/model.pth'))

model.eval()
with torch.inference_mode():
    y_pred = model(X_test)
    test_loss = loss_fn(y_pred , y_test)

print(f"test_loss : {test_loss}")    
