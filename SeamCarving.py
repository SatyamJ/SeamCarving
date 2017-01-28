__author__ = 'Satyam'

import pylab
import math
from skimage import img_as_float


class SeamCarving:

    def dual_gradient_energy(self, img_path):
        """
        Input: Path of the image
        Output:
        a W x H array of floats, the energy at each pixel in input img.
        W is width of the image
        H is height of the image
        Working:
        Slices the image into its R, G and B components.
        Calculates energy of each pixel by dual gradient energy function.
        """

        img = pylab.imread(img_path)
        img = img_as_float(img)
        height, width = img.shape[:2]
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        energy = [[-1 for i in range(width)] for j in range(height)]

        for i in range(height):
            for j in range(width):
                if i == 0:
                    ry = r[i+1][j] - r[height-1][j]
                    gy = g[i+1][j] - g[height-1][j]
                    by = b[i+1][j] - b[height-1][j]
                elif i == height - 1:
                    ry = r[0][j] - r[i-1][j]
                    gy = g[0][j] - g[i-1][j]
                    by = b[0][j] - b[i-1][j]
                else:
                    ry = r[i+1][j] - r[i-1][j]
                    gy = g[i+1][j] - g[i-1][j]
                    by = b[i+1][j] - b[i-1][j]

                if j == 0:
                    rx = r[i][j+1] - r[i][width-1]
                    gx = g[i][j+1] - g[i][width-1]
                    bx = b[i][j+1] - b[i][width-1]
                elif j == width - 1:
                    rx = r[i][0] - r[i][j-1]
                    gx = g[i][0] - g[i][j-1]
                    bx = b[i][0] - b[i][j-1]
                else:
                    rx = r[i][j+1] - r[i][j-1]
                    gx = g[i][j+1] - g[i][j-1]
                    bx = b[i][j+1] - b[i][j-1]

                delta_x = math.pow(rx, 2) + math.pow(gx, 2) + math.pow(bx, 2)
                delta_y = math.pow(ry, 2) + math.pow(gy, 2) + math.pow(by, 2)

                energy[i][j] = delta_x + delta_y

        return energy

    def find_seam(self, energy):
        """
        Input: Array of energy of each pixel
        Output:
        1. A List having column numbers for each row of the image,
        depicting the seam with lowest cost.
        2. Cost of the seam found with lowest cost.
        Working:
        Performs memoization by saving the solution of sub-problems in the
        form of matrices cost and path.
        Cost matrix helps us to find minimum cost at iteration without
        recomputing it.
        Path matrix helps us to find the seam co-ordinates without
        recomputing it.
        """

        energy = img_as_float(energy)
        height, width = energy.shape

        cost = [[0 for i in range(width)] for j in range(height)]
        path = [[-1 for i in range(width)] for j in range(height)]

        for i in range(width):
            cost[height-1][i] = energy[height-1][i]

        for i in range(height-2, -1, -1):
            for j in range(width):
                if j == 0:
                    adj = [cost[i+1][j], cost[i+1][j+1]]
                    cost[i][j] = energy[i][j] + min(adj)
                    path[i][j] = j + adj.index(min(adj))
                elif j == width-1:
                    adj = [cost[i+1][j-1], cost[i+1][j]]
                    cost[i][j] = energy[i][j] + min(adj)
                    path[i][j] = j + adj.index(min(adj)) - 1
                else:
                    adj = [cost[i+1][j-1], cost[i+1][j], cost[i+1][j+1]]
                    cost[i][j] = energy[i][j] + min(adj)
                    path[i][j] = j + adj.index(min(adj)) - 1

        min_cost = cost[0][0]
        start_index = 0
        for i in range(width):
            if cost[0][i] < min_cost:
                min_cost = cost[0][i]
                start_index = i

        seam = [start_index]
        next_index = start_index

        for i in range(height):
            seam = seam + [path[i][next_index]]
            next_index = path[i][next_index]
        return seam, min_cost

    def plot_seam(self, image, seam):
        """
        Input:
        1. image: Original image in the first instance and then
        reduced image in subsequent iteration
        2. seam: list having column numbers for each row of the image,
        depicting the seam to be removed.
        Output:
        Input image with the seam drawn, showing the seam visualization
        Working:
        Replaces the RGB values of the seam pixel co-ordinates identified by
        list - seam with (0.7, 0, 0) which RGB value for red
        """

        seam_plot = pylab.imread(image)
        seam_plot = img_as_float(seam_plot)

        height, width = seam_plot.shape[0:2]

        for i in range(height):
            for j in range(width):
                if seam[i] == j:
                    seam_plot[i][j][0] = 0.7
                    seam_plot[i][j][1] = 0
                    seam_plot[i][j][2] = 0
        pylab.imsave("SeamPlot", seam_plot)
        return seam_plot

    def remove_seam(self, img, seam):
        """
        Input:
        1. img: Original image in the first instance and then
        reduced image in subsequent iteration
        2. seam: list having column numbers for each row of the image,
        depicting the seam to be removed.
        Output:
        NewImage.png: New image saved having the seam pixels removed
        new_img: New image having the seam pixels removed from input image
        Working:
        We copy the non-seam pixels from img to a new image.
        We adopt this method since nd array is immutable.
        """

        img = pylab.imread(img)
        img = img_as_float(img)

        height = img.shape[0]
        width = img.shape[1]

        new_img = [[[0 for k in range(3)] for i in range(width-1)]
                   for j in range(height)]

        new_img = img_as_float(new_img)

        y = 0
        for i in range(height):
            for j in range(width):
                if j != seam[i]:
                    new_img[i][y][0] = img[i][j][0]
                    new_img[i][y][1] = img[i][j][1]
                    new_img[i][y][2] = img[i][j][2]
                    y = (y + 1) % (width - 1)

        pylab.imsave('NewImage.png', new_img)
        return new_img


if __name__ == '__main__':

    sc = SeamCarving()

    img = pylab.imread("HJoceanSmall.png")
    img = img_as_float(img)

    pylab.figure()
    pylab.gray()

    # Plotting the input image
    pylab.subplot(2, 2, 1)
    pylab.imshow(img)
    pylab.title('Original image')

    # Plotting the energy function of the image
    energy_image = sc.dual_gradient_energy('HJoceanSmall.png')
    pylab.subplot(2, 2, 2)
    pylab.imshow(energy_image)
    pylab.title('Energy function plot')

    seam, mininum_cost = sc.find_seam(energy_image)
    print "Iteration 1: Cost of seam to be removed: ", mininum_cost

    # Plotting the seam found on the input image
    seam_plot = sc.plot_seam('HJoceanSmall.png', seam)
    pylab.subplot(2, 2, 3)
    pylab.imshow(seam_plot)
    pylab.title('Seam Plot')

    new_image = sc.remove_seam('HJoceanSmall.png', seam)

    # Iterating the entire process 100 times to see significant reduction.
    for i in range(100):
        energy_image = sc.dual_gradient_energy('NewImage.png')
        seam, mininum_cost = sc.find_seam(energy_image)
        print "Iteration", (i+2), ": Cost of seam to be removed: ",\
            mininum_cost
        seam_plot = sc.plot_seam('SeamPlot.png', seam)
        new_image = sc.remove_seam('NewImage.png', seam)

    # Plotting the final image after computation
    pylab.subplot(2, 2, 4)
    pylab.imshow(new_image)
    pylab.title('Final Image')
    pylab.show()
