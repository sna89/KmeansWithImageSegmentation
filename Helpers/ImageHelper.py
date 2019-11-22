import matplotlib.pyplot as plt

class ImageHelper():
    def __init__(self):
        pass

    @staticmethod
    def show_image(image):
        plt.figure()
        plt.imshow(image)
        plt.show()
