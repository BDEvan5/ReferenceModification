import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class NavGenertor:
    def __init__(self, map_name, width, height):
        self.name = map_name
        self.height = height
        self.width = width
        self.resolution = 0.05

        self.img_height = int(height/self.resolution)
        self.img_width = int(width/self.resolution)

        self.map_img = np.zeros((self.img_width, self.img_height))

    def xy_to_row_column(self, pt):
        c = int(round(np.clip(pt[0] / self.resolution, 0, self.img_width-2)))
        r = int(round(np.clip(pt[1] / self.resolution, 0, self.img_height-2)))
        return c, r


    def add_circle(self, location, radius):
        x, y = self.xy_to_row_column(location)

        r = int(radius/self.resolution)

        locations = []
        for i in range(-r, r):
            for j in range(-r, r):
                if np.sqrt(i**2 + j**2) < r:
                    locations.append([i, j])

        img_mask = np.zeros((self.img_width, self.img_height))
        for location in locations:
            img_mask[location[0]+x, location[1]+y] = 1

        self.map_img = img_mask + self.map_img

    def show_map(self, wait=True):
        plt.figure(1)
        plt.imshow(self.map_img)
        plt.show()

    def add_rectangle(self, location, size):
        x, y = self.xy_to_row_column(location)
        sx, sy = self.xy_to_row_column(size)

        img_mask = np.zeros((self.img_width, self.img_height))
        for i in range(sx):
            for j in range(sy):
                img_mask[i+x, j+y] = 1
        self.map_img = img_mask + self.map_img

    def save_map(self, path):
        img_path = path + self.name + '.png'
        img_arr = np.array(self.map_img, np.uint8) * 255
        img = Image.fromarray(img_arr)
        img.show()
        img.save(img_path)
        # plt.figure(1)
        # plt.imshow(self.map_img)
        # plt.savefig(img_path)
        
        print(f"Saved img: {img_path}")

def make_pfeiffer():


    my_map = NavGenertor("pfeiffer", 10, 10)
    my_map.add_circle([7, 8], 0.8)
    my_map.add_rectangle([2, 4.25], [5, 1.5])

    my_map.save_map("nav_maps/")

    # my_map.show_map()


if __name__ == "__main__":
    make_pfeiffer()
    # main()