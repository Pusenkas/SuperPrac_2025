import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


K = 10 + 1
N = 16
tau = 0.1
shape = (K, N + 1, N + 1, N + 1)
layer2d = 3

def create_plots(name):
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
    print(f"MAX {max(arr_1d)} MIN {min(arr_1d)}")
                

    # # Sample data
    # data = np.random.rand(10, 10) * 100

    # Create a parameterized heatmap
    for i in range(K):
        data_3d = data[i]
        X, Y, Z = np.indices(data_3d.shape)
    
        # Flatten all arrays
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        data_flat = data_3d.flatten()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Use a colormap to map data values to colors
        cmap = plt.get_cmap('coolwarm')
        # Normalize data values to range [0, 1] for the colormap
        norm = plt.Normalize(vmin=data_flat.min(), vmax=data_flat.max())
        colors = cmap(norm(data_flat))

        # Plot as a 3D scatter
        scatter = ax.scatter(x_flat, y_flat, z_flat, c=data_flat, marker='o', s=10, cmap='coolwarm', vmin=min(arr_1d), vmax=max(arr_1d)) # Use square markers

        ax.set_title(f"Plot at t = {i * tau:.1f}")
        ax.set_xlabel('X index')
        ax.set_ylabel('Y index')
        ax.set_zlabel('Z index')

        # Add a color bar
        fig.colorbar(scatter, ax=ax, pad=0.1)
        
        plt.savefig(f'data/{name}_{i}.png')
        # plt.show()

def create_gif_from_images(name, duration=100):
    images = []
    for t in range(K):
        img = Image.open(f"data/{name}_{t}.png")
        images.append(img)

    if not images:
        print("No valid images found to create GIF.")
        return

    # Save the first image as the starting frame, then append the rest
    images[0].save(
        f"data/{name}.gif",
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0  # 0 means infinite loop
    )
    print(f"GIF created successfully at data/plots.gif")

if __name__ == "__main__":
    name = "rel_error_plot"
    create_plots(name)
    create_gif_from_images(name, 1000)