import numpy as np
from keras.models import Model, load_model
from PIL import Image

model = load_model("generator_model.h5")
model.load_weights('generator_weights.h5')
z_dim = 1000
CLASS_NUM = 10
REPEAT = 100

def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = 10
    rows = 10
    WIDTH, HEIGHT = generated_images.shape[1:3]
    combined_image = np.zeros((HEIGHT*rows, WIDTH*cols),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = int(index % cols)
        combined_image[WIDTH*i:WIDTH*(i+1), HEIGHT*j:HEIGHT*(j+1)] = image[:, :, 0]

    return combined_image

def draw_image(one_hot, filename='img'):
    noise = np.random.uniform(-1, 1, [REPEAT, z_dim])
    one_hot_repeat = np.tile(one_hot, REPEAT).reshape(REPEAT, CLASS_NUM)
    noise_with_one_hot = np.concatenate([noise, one_hot_repeat], axis=1)
    generated_images = model.predict(noise_with_one_hot)
    img = combine_images(generated_images)
    img = img*127.5 + 127.5
    Image.fromarray(img.astype(np.uint8)).save("%s.png" % (filename))

def test1():
    label = [3]
    one_hot = np.eye(CLASS_NUM)[label].reshape(CLASS_NUM)
    draw_image(one_hot, "draw3")

if __name__ == '__main__':
    test1()


