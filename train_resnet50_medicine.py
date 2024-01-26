import os
import math
import argparse
from my_dataset_medicine_for_test import MyDataSet_for_test
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import random
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from my_dataset_medicine import MyDataSet
from utils_medicine import train_one_epoch, evaluate
import warnings
import datetime

warnings.filterwarnings("ignore")


def main(args):
    random.seed(0)
    # Check whether GPU is supported, if so use GPU, otherwise use CPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("torch.cuda.is_available:", device)
    # 如果文件夹 "./weights" 不存在，创建它
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    labelclass = {"CSR": 1, "HC": 0}

    # Defines the way data is transformed for data preprocessing
    data_transform = {
        "train": transforms.Compose([transforms.Resize((512, 512))]),
        "val": transforms.Compose([transforms.Resize((512, 512))]),
    }

    # Instantiate the training dataset
    train_dataset = MyDataSet(
        mode="train", transform=data_transform["train"], labelclass=labelclass
    )

    # Instantiate the validation dataset
    val_dataset = MyDataSet(
        mode="val", transform=data_transform["val"], labelclass=labelclass
    )
    # Instantiate the test dataset
    test_dataset = MyDataSet_for_test(
        mode="val", transform=data_transform["val"], labelclass=labelclass
    )
    # Set the parameters for the batch size and the data loader
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    # Create training and validation data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,  # batch size
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )
    # The ResNet architecture was used to create the model and set the parameters of the fully connected and convolutional layers
    model = models.resnet50().to(device)
    model.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), bias=False).to(
        device
    )
    model.fc = nn.Linear(2048, args.num_classes).to(device)

    # If the weight file for the pre-trained model is provided, load the weight
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(
            args.weights
        )
        weights_dict = torch.load(args.weights)
        print(
            "this is model.load_state_dict:",
            model.load_state_dict(weights_dict, strict=False),
        )

    # Get the parameters that need to be optimized, set the optimizer and the learning rate scheduler
    pg = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.SGD(pg, lr=args.lr, momentum=1, weight_decay=5e-5)
    lf = (
        lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf)
        + args.lrf
    )

    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lf
    )  #  This scheduler dynamically updates the learning rate in the optimizer according to the process in each epoch to help the model converge better

    for epoch in range(args.epochs):
        y_true = []
        y_score = []
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            y_true=y_true,
            y_score=y_score,
        )
        # Update learning rate
        scheduler.step()

        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch,
            y_true=y_true,
            y_score=y_score,
        )
        test_loss, test_acc = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
            epoch=epoch,
            y_true=y_true,
            y_score=y_score,
        )
        print("val_loss, val_acc is ", val_loss, val_acc)
        print("train_loss, train_acc is ", train_loss, train_acc)
        with open("./weights/information50.txt", "a") as f:
            f.writelines(
                str(datetime.datetime.now())
                + "_"
                + str(epoch)
                + "_"
                + "val_loss, val_acc is "
                + str(val_loss)
                + "_"
                + str(val_acc)
                + "train_loss, train_acc is "
                + str(train_loss)
                + "_"
                + str(train_acc)
                + "test_loss, test_acc is "
                + str(test_loss)
                + "_"
                + str(test_acc)
                + "\n"
            )
        torch.save(model.state_dict(), "./weights/model_resnet50-epoch{}.pth".format(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument(
        "--model-name",
        default="",
        help="create model name",
    )
    # Pretrain weight paths, set to null characters if you don't want to load them
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="initial weights path",
    )
    parser.add_argument("--freeze-layers", type=bool, default=True)
    parser.add_argument(
        "--device", default="cuda:0", help="device id (i.e. 0 or 0,1 or cpu)"
    )
    opt = parser.parse_args()
    main(opt)

