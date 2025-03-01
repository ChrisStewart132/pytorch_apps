import argparse
import torchvision.transforms.v2 as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from model import ImageToImageCNN
from loss import VGGPerceptualLoss, WeightedMSELoss, MaxPoolLoss, EffNetV2PerceptualLoss, ConvNeXtTinyPerceptualLoss
from dataset import PairedImageDataset
from tensorboard_logger import TensorboardLogger
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Train an image-to-image translation model.')
    parser.add_argument('--n_features', type=int, default=16, help='Number of features in the model')
    parser.add_argument('--starting_epoch', type=int, default=0, help='Starting epoch number')
    parser.add_argument('--n_epochs', type=int, default=12, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for AdamW optimizer')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--scale_factor', type=int, default=2, help='Scale factor for model')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = TensorboardLogger()

    transform_train = transforms.Compose([
        transforms.ToImage(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToDtype(torch.float32, scale=True)
    ])

    dataset = PairedImageDataset('data/train', transform=transform_train)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    model_path = f'models/model_{args.n_features}f_v1.pth'
    model = ImageToImageCNN(3, 3, features=args.n_features, scale_factor=args.scale_factor)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Creating new model at {model_path}")

    #criterion = VGGPerceptualLoss(False, device, feature_layers=[], style_layers=[0, 1], style_weights=[0.75, 0.25])
    #criterion = EffNetV2PerceptualLoss(False, device)
    criterion = ConvNeXtTinyPerceptualLoss(False, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model.to(device)
    best_loss = float('inf')

    for epoch in range(args.starting_epoch, args.starting_epoch + args.n_epochs):
        for step, (input_images, target_images) in enumerate(dataloader):
            valid_indices = [j for j, (inp, tar) in enumerate(zip(input_images, target_images)) if inp is not None and tar is not None]

            if not valid_indices:
                continue

            input_images = torch.stack([input_images[i] for i in valid_indices]).to(device)
            target_images = torch.stack([target_images[i] for i in valid_indices]).to(device)

            outputs = model(input_images)

            if step == 0:
                input_grid = vutils.make_grid(input_images[:1].detach().cpu())
                output_grid = vutils.make_grid(outputs[:1].detach().cpu())
                target_grid = vutils.make_grid(target_images[:1].detach().cpu())
                combined_grid = torch.cat([input_grid, target_grid, output_grid], dim=2)
                logger.log_image(combined_grid, tag=f"Combined Images", dataformats="CHW", step=epoch)

            loss = criterion(outputs, target_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if loss.item() < best_loss:
            torch.save(model.state_dict(), model_path)
            best_loss = loss.item()

        #torch.save(model.state_dict(), model_path[:-4] + f'_{epoch}.pth')
        if loss < 1e-4:
            print(f"Epoch [{epoch+1}/{args.starting_epoch + args.n_epochs}], Loss: {loss.item()}, Best Loss: {best_loss}")
        else:
            print(f"Epoch [{epoch+1}/{args.starting_epoch + args.n_epochs}], Loss: {loss.item():.4f}, Best Loss: {best_loss:.4f}")
            
        logger.log_epoch_loss(epoch, loss.item(), best_loss)

    dataset.close()


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
    print("training finished")
