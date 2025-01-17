from src.core.data.data_loader import DataLoader

data_loader = DataLoader()

# must load final data here
customer_reviews = data_loader.get_customer_reviews("final")
train, test = data_loader.get_train_test_split(customer_reviews)

import os
from typing import Tuple

import torch
import torch.nn as nn
# define the network
class Customer_Review_NN(nn.Module):
     def __init__(self, input_size=300, hidden_size=100, num_classes=4):
        super(Customer_Review_NN, self).__init__()

        self.layer_1 = nn.Linear(input_size,hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)
    
    # accept input and return an output
     def forward(self, x):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out
     
def get_loss_fn_and_optimizer(review_net: Customer_Review_NN, learning_rate: float) -> Tuple:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(review_net.parameters(), lr=learning_rate) 
    return loss_fn, optimizer 

# Train the Model
def train(model, optimizer, loss_fn, train_data, num_epochs=1, batch_size=8, save_path=None, count_vectorizer=None, device="cpu"):
    total_loss = 0

    for epoch in range(num_epochs):
        
        # determine the number of min-batches based on the batch size and size of training data
        total_batch = int(len(train_data)/batch_size)
        
        # Loop over all batches
        for i in range(total_batch):

            batch_x, batch_y = get_batch(train_data,i,batch_size, train_data['CATEGORY_INDEX'].unique(), count_vectorizer)
            articles = torch.FloatTensor(batch_x).to(device)
            labels = torch.LongTensor(batch_y).to(device)
            # necessary for BCEWithLogitsLoss
            # if len(train_data['CATEGORY_INDEX'].unique()) == 2:
            #     labels = torch.LongTensor(batch_y).unsqueeze(1).float().to(device)
                
            optimizer.zero_grad()  # zero the gradient buffer
            # Forward + Backward + Optimize
            outputs = model(articles)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()


            if (i+1) % 4 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                       %(epoch+1, num_epochs, i+1, 
                         len(train_data)/batch_size, loss.data))
                
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"saving model to {save_path}")
        torch.save(model.state_dict(), f"{save_path}")


    return model  
