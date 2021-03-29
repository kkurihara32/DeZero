import numpy as np

from PIL import Image
from dezero.models import VGG16
import dezero


def main():
    url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
    img_path = dezero.utils.get_file(url)
    img = Image.open(img_path)
    x = VGG16.preprocess(img)
    print(type(x), x.shape)
    print("-"*30)

    img_path = dezero.utils.get_file(url)
    img = Image.open(img_path)
    x = VGG16.preprocess(img)
    print(x.shape)
    print(x)
    print("-"*30)

    x = x[np.newaxis]

    print(x.shape)
    print(x)
    print("-" * 30)

    model = VGG16(pretrained=True)
    with dezero.test_mode():
        y = model(x)
        print(y.shape)
    predict_id = np.argmax(y.data)

    labels = dezero.datasets.ImageNet.labels()
    print(labels[predict_id])


if __name__ == "__main__":
    main()
