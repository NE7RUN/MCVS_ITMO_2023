import sys
import getopt
import time
import torch
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models.alexnet import alexnet
from torch2trt import torch2trt
from torch2trt import TRTModule

# Classes load
with open('./config/classes.csv', 'r') as fd:
    dc = csv.DictReader(fd)
    classes = {}
    for line in dc:
        classes[int(line['class_id'])] = line['class_name']

# Defining a device to work with PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for input images
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

# Output directory 
Path("./output").mkdir(exist_ok=True)

def process_images(images: list,
                   trt: bool):
    timest = time.time()
    if trt:                                                   # Defining trt for opt
        x = torch.ones((1, 3, 224, 224)).cuda()
        model = alexnet(pretrained=True).eval().cuda()
        model_trt = torch2trt(model, [x])
        torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
        model = model_trt
        model = TRTModule()
        model.load_state_dict(torch.load('alexnet_trt.pth'))
    else:                                                       # usual AlexNet
        model = alexnet(pretrained=True).eval().cuda()
    print("Model load time {}".format(time.time() - timest))

    # Image classification using the model
    timest = time.time()
    for image in images:
        index = classify_image(image, model)
        output_text = str(index) + ': ' + classes[index]
        # Output image edit
        edit = ImageDraw.Draw(image)
        edit.rectangle((0, image.height - 20, image.width, image.height),
                       fill=(255, 255, 255))
        edit.text((50, image.height-15), output_text, (0, 0, 0),
                  font=ImageFont.load_default())
        image.save('./output/' + image.filename.split('/')[-1])

    print("Image(s) processing time {}".format(time.time() - timest))
    print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
    print('Max memory allocated: ' + str(torch.cuda.max_memory_allocated()))

# model go brrrrr
def classify_image(image: Image,
                   model) -> int:
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor).to(device)

    #class index return
    output = model(input)
    return output.data.cpu().numpy().argmax()


def print_usage():
    print("Usage: python lab3.py --trt /path/to/images/directory")


def main(argv: list, trt: bool = False):
    try:
        opts, _ = getopt.getopt(argv, "", ["trt"])
        if len(opts) == 1:
            trt = True
            argv.remove('--trt')
        elif len(opts) > 1:
            raise getopt.GetoptError("invalid arguments")
    except getopt.GetoptError:
        print_usage()
        sys.exit(1)

    # Get the path to the directory from the command line arguments
    if argv:
        directory_path = argv[0]
    else:
        print_usage()
        sys.exit(1)

    if not Path(directory_path).is_dir():
        print(f"Error: {directory_path} is not a directory.")
        sys.exit(1)

    # Get files
    image_files = [f for f in Path(directory_path).glob("*.jpg")]

    if not image_files:
        print(f"Error: No JPG images found in {directory_path}.")
        sys.exit(1)

    # make list
    images = [Image.open(img_path) for img_path in image_files]

    process_images(images, trt)

if __name__ == "__main__":
    main(sys.argv[1:])