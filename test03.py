# Author:		Kwangryeol Park
# Email:		pkr7098@gmail.com
# File name:	test01.py
# Repo:			https://github.com/KwangryeolPark
# Created on:	Tue Sep 19 2023
# Modified on:	Tue Sep 19 2023 12:41:29 PM
# Description:	
#
# Copyright (c) 2023 Kwangryeol Park All Rights Reserved.
#
# PyTorch 문서에 따르면, BatchNorm 후에 나오는 Conv에 대해 bias가 필요 없다고 합니다.
# 즉, 다음과 같은 구조에서 Conv의 bias는 False로 설정해도 무방합니다.
# https://github.com/KwangryeolPark/PyTorch.Improve.Speed.git#

# batch_size = 128

import torch
import torch.nn as nn
import pandas as pd
import argparse
from torchvision import datasets, transforms
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--torch_no_grad", type=bool, required=True)
args = parser.parse_args()

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.bias = True
        self.model = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 64, 3, 1, 1, bias=self.bias),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1, bias=self.bias),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1, bias=self.bias),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1, bias=self.bias),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1, bias=self.bias),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, 3, 1, 1, bias=self.bias),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(28*28, 10)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
    
def train(model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def val(model, val_loader):
    model.eval()
    if args.torch_no_grad:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data = data.cuda()
                target = target.cuda()
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
    else:
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
        

if __name__ == "__main__":
    import time
    
####################################################################################################
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    val_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False)
    
    model = TestNet(mode=args.biased).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    tqdm_loader = tqdm(range(20))
    start = time.time()
    for epoch in tqdm_loader:
        train(model, optimizer, train_loader)
        val(model, val_loader)
        
    elapsed_time = time.time() - start
    print("elapsed time: ", elapsed_time)
    with open(f"docs/test03_torch_no_grad{args.torch_no_grad}.txt", "a") as f:
        f.write(str(elapsed_time) + "\n")
    
####################################################################################################
