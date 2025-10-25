import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


K = 10 + 1
N = 16
tau = 0.1
shape = (K, N + 1, N + 1, N + 1)
layer2d = 3

def create_plots():
    with open('test.data', 'r') as f:
        line = f.readline()
        
    numbers_str = line.strip().split()
    numbers = [float(num) for num in numbers_str]

    arr_1d = np.array(numbers)

    if arr_1d.size != np.prod(shape):
        raise ValueError(f"Number of elements in the file ({arr_1d.size}) does not match "
                            f"the product of dimensions in the target shape ({np.prod(shape)}).")

    data = arr_1d.reshape(shape)
    print(data[0][0].shape)
                

    # # Sample data
    # data = np.random.rand(10, 10) * 100

    # Create a parameterized heatmap
    for i in range(K):
        plt.figure(figsize=(8, 8))
        sns.heatmap(data[i][layer2d],
                    cmap='coolwarm',      # Colormap
                    annot=True,          # Show annotations
                    fmt=".1f",           # Format annotations to one decimal place
                    linewidths=0.5,      # Add lines between cells
                    linecolor='black',   # Color of the lines
                    cbar=True)           # Display colorbar
        plt.title(f"Plot at i = {layer2d}; t = {i * tau:.1f}")
        plt.savefig(f'data/plot_{i}.png')
        plt.close()
        print(f"Plot data/plot_{i}.png created")

def create_gif_from_images(duration=100):
        images = []
        for t in range(K):
            img = Image.open(f"data/plot_{t}.png")
            images.append(img)

        if not images:
            print("No valid images found to create GIF.")
            return

        # Save the first image as the starting frame, then append the rest
        images[0].save(
            "data/plots.gif",
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0  # 0 means infinite loop
        )
        print(f"GIF created successfully at data/plots.gif")

if __name__ == "__main__":
    create_plots()
    create_gif_from_images(1000)