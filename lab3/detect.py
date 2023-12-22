import sys
import getopt
import time
import torch
import csv
import glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models.alexnet import alexnet
from torch2trt import torch2trt
from torch2trt import TRTModule

# Classes load
with open('./config/classes.csv', 'r') as fd:
    dev = csv.DictReader(fd)
    classes = {}
    for line in dev:
        classes[int(line['class_id'])] = line['class_name']

# Defining a device to work with PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for input images
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

# Output directory 
Path("./output").mkdir(exist_ok=True)
images = []
def image_processing(images: list,
                   trt: bool):
    times = time.time()
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
    print("Model load time {}".format(time.time() - times))

    # Image classification using the model
    times = time.time()
    for image in images:
        index = image_classification(image, model)
        output_text = str(index) + ': ' + classes[index]
        # Output image edit
        edit = ImageDraw.Draw(image)
        edit.rectangle((0, image.height - 20, image.width, image.height),
                       fill=(255, 255, 255))
        edit.text((50, image.height-15), output_text, (0, 0, 0),
                  font=ImageFont.load_default())
        image.save('./output/' + image.filename.split('/')[-1])
    print(images)
    print("Image(s) processing time {}".format(time.time() - times))
    print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
    print('Max memory allocated: ' + str(torch.cuda.max_memory_allocated()))

# model go brrrrr
def image_classification(image: Image,
                   model) -> int:
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor).to(device)
    #class index return
    output = model(input)
    return output.data.cpu().numpy().argmax()


def print_usage():
    print("Usage: python detect.py --trt")

def main(argv: list,
         trt: bool = False):
    try:
        opts, empty = getopt.getopt(argv, "", ["trt"])
        if len(opts) == 1:
            trt = True
            argv.remove('--trt')
        elif len(opts) > 1:
            raise getopt.GetoptError("is not a directory.")
    except getopt.GetoptError:
        print_usage()
        sys.exit(1)

    for file in glob.glob('data/*.jpg', recursive=True):
        try:
            image = Image.open(file)
            images.append(image)
        except FileNotFoundError:
            print(file + " not found")

    if len(images) == 0:
        print_usage()
        sys.exit(1)

    image_processing(images, trt)


if __name__ == "__main__":
    main(sys.argv[1:])
