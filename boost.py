import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

thread_count_128 = [1, 2, 4, 8, 18]
thread_count_256 = [2, 4, 8, 16, 32]

time_128 = [31.28, 15.76, 8.00, 4.09, 2.89]
time_256 = [125.55, 65.48, 32.52, 21.94, 15.96]

speedup_128 = [time_128[0] / t for t in time_128]
speedup_256 = [time_256[0] / t for t in time_256]

def time():
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(thread_count_128, time_128, 'o-', label='N = 128')
    axes[0].set_title('Execution Time vs Thread Count')
    axes[0].set_xlabel('Thread Count')
    axes[0].set_ylabel('Execution Time (s)')
    axes[0].legend()
    axes[0].set_xticks(thread_count_128)

    axes[1].plot(thread_count_256, time_256, 's-', color='orange', label='N = 256')
    axes[1].set_title('Execution Time vs Thread Count')
    axes[1].set_xlabel('Thread Count')
    axes[1].set_ylabel('Execution Time')
    axes[1].legend()
    axes[1].set_xticks(thread_count_256)

    fig.suptitle('Performance Comparison', fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig('data/performance_plots.png', dpi=300)

    print("The combined plot has been saved as 'data/performance_plots.png'")
    
def boost():
    # Plotting the speedup results
    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the calculated speedup for dataset 128
    ax.plot(thread_count_128, speedup_128, 'o-', label='N = 128', color='blue')

    # Plot the calculated speedup for dataset 256
    ax.plot(thread_count_256, speedup_256, 's-', label='N = 256', color='orange')

    # Plot the ideal speedup line for comparison
    # Ideal speedup is when speedup equals the number of threads
    max_threads = max(max(thread_count_128), max(thread_count_256))
    ideal_speedup = np.arange(1, max_threads + 2)
    ax.plot(ideal_speedup, ideal_speedup, '--', label='Ideal Speedup', color='gray', alpha=0.7)

    # Customize the plot
    ax.set_title('Relative Speedup of Execution Time', fontsize=16)
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.legend()
    ax.set_xticks(np.unique(thread_count_128 + thread_count_256))
    ax.set_yticks(np.arange(0, max(speedup_128 + speedup_256) + 2, 2))
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.set_yscale('log', base=2)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%dx"))


    # Add grid lines
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Save the figure as a high-quality PNG
    plt.tight_layout()
    plt.savefig('data/relative_speedup_plot.png', dpi=300)

    print("The relative speedup plot has been saved as 'data/relative_speedup_plot.png'")
    
time()
boost()