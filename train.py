import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torchvision import transforms
import torch.optim as optim
from PIL import Image
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

from Model import AdaIN_net, encoder_decoder
from sampler import InfiniteSamplerWrapper


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = [p for p in Path(self.root).glob('*') if p.is_file() and p.suffix in {'.jpg', '.jpeg', '.png'}]
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def main():
    cudnn.benchmark = True
    Image.MAX_IMAGE_PIXELS = None

    parser = argparse.ArgumentParser(description='Style Transfer')
    parser.add_argument('-content_dir', type=str, required=True, help='Path to content image directory')
    parser.add_argument('-style_dir', type=str, required=True, help='Path to style image directory')
    parser.add_argument('-gamma', type=float, default=10.0, help='Weight for the style loss')
    parser.add_argument('-e', type=int, default=20, help='Number of epochs')
    parser.add_argument('-b', type=int, default=8, help='Batch size')
    parser.add_argument('-l', type=str, required=True, help='Path to encoder weights file')
    parser.add_argument('-s', type=str, required=True, help='Path to decoder weights file')
    parser.add_argument('-cuda', type=str, default='Y', help='Enable CUDA Y/N')

    args = parser.parse_args()

    content_data_path = args.content_dir
    style_data_path = args.style_dir
    gamma = args.gamma
    epochs = args.e
    batch_size = args.b
    encoder_weights_path = args.l
    decoder_weights_path = args.s
    use_cuda = args.cuda.upper() == 'Y'


    def train_transform():
        transform_list = [
            transforms.Resize(size=(512, 512)),
            transforms.RandomCrop(256),
            transforms.ToTensor()
        ]
        return transforms.Compose(transform_list)

    learning_rate = 0.001
    threads = 6

    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    decoder_weights_path = Path(decoder_weights_path)
    decoder_weights_path.mkdir(exist_ok=True, parents=True)
    # intilize the model
    encoder = encoder_decoder.encoder
    decoder = encoder_decoder.decoder

    encoder.load_state_dict(torch.load(encoder_weights_path))

    model = AdaIN_net(encoder, decoder)
    model.train()
    model.to(device)

    content_transform = train_transform()
    style_transform = train_transform()

    content_dataset = FlatFolderDataset(content_data_path, content_transform)
    style_dataset = FlatFolderDataset(style_data_path, style_transform)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=threads))

    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=threads))

    optimizer = optim.Adam(model.decoder.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False)

    print('Training...')

    content_losses = []
    style_losses = []
    total_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in range(batch_size):
            content_images = next(content_iter).to(device)
            style_images = next(style_iter).to(device)

            content_images = content_images.unsqueeze(0)
            style_images = style_images.unsqueeze(0)

            for content_image, style_image in zip(content_images, style_images):
                loss_c, loss_s = model(content_image, style_image)
                loss_c = 1.0 * loss_c
                loss_s = gamma * loss_s
                loss = loss_c + loss_s
                total_loss += loss.item()

                # Backpropagation through the network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        avg_loss = total_loss / batch_size
        scheduler.step(avg_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss}')

        content_losses.append(loss_c.item())
        style_losses.append(loss_s.item())
        total_losses.append(loss.item())

    decoder_weights_folder = decoder_weights_path / "decoder_weights"
    decoder_weights_folder.mkdir(exist_ok=True, parents=True)
    decoder_weights_path = decoder_weights_folder / "decoder_weights.pth"
    torch.save(model.decoder.state_dict(), str(decoder_weights_path))

    plt.plot(range(len(content_losses)), content_losses, label='Content Loss')
    plt.plot(range(len(style_losses)), style_losses, label='Style Loss')
    plt.plot(range(len(total_losses)), total_losses, label='Total Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
