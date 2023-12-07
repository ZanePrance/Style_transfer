import torch
from torchvision import transforms
from PIL import Image
import argparse

from Model import AdaIN_net, encoder_decoder

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def style_transfer(content_img, style_img, model):
    with torch.no_grad():
        stylized_img = model(content_img, style_img, alpha=0.1)

    return stylized_img

def save_img(img,output_path):
    img = img.squeeze(0).cpu().clamp(0, 1)
    img = transforms.ToPILImage()(img)
    img.save(output_path, "JPEG")

def main(args):
    use_cuda = args.cuda.upper() == 'Y'
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    # intilize the model
    encoder = encoder_decoder.encoder
    decoder = encoder_decoder.decoder

    encoder_weights_path = args.encoder_weights
    encoder.load_state_dict(torch.load(encoder_weights_path))

    decoder_weights_path = args.decoder_weights
    decoder.load_state_dict(torch.load(decoder_weights_path, map_location='cpu'))

    model = AdaIN_net(encoder, decoder)
    model.eval()
    model.to(device)

    content_image_path = args.content_image
    style_image_path = args.style_image
    output_image_path = args.output_image

    content_image = process_image(content_image_path)
    style_image = process_image(style_image_path)

    stylized_image = style_transfer(content_image, style_image, model)

    save_img(stylized_image, output_image_path)

    print(f"Stylized image saved to {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Style Transfer')
    parser.add_argument('-content_image', type=str, required=True, help='Path to content image')
    parser.add_argument('-style_image', type=str, required=True, help='Path to style image')
    parser.add_argument('-encoder_weights', type=str, required=True, help='Path to encoder.pth file')
    parser.add_argument('-decoder_weights', type=str, required=True, help='Path to decoder.pth file')
    parser.add_argument('-output_image', type=str, required=True, help='Path to stylized output image')
    parser.add_argument('-alpha', type=float, required=True, help='Alpha value')
    parser.add_argument('-cuda', type=str, default='Y', help='Enable CUDA Y/N')

    args =parser.parse_args()
    main(args)
