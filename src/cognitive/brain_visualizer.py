import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class BrainVisualizer:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.fig, self.axes = plt.subplots(1, len(layer_sizes), figsize=(4*len(layer_sizes), 4))
        if len(layer_sizes) == 1:
            self.axes = [self.axes]
        self.scatters = []
        for ax, size in zip(self.axes, layer_sizes):
            n = int(np.ceil(np.sqrt(size)))
            x, y = np.meshgrid(np.arange(n), np.arange(n))
            coords = np.stack([x.flatten(), y.flatten()], axis=1)[:size]
            scatter = ax.scatter(coords[:,0], coords[:,1], c='black', s=60, alpha=0.2)
            ax.set_title(f'Layer ({size} neurons)')
            ax.set_xticks([])
            ax.set_yticks([])
            self.scatters.append(scatter)
        self.fig.tight_layout()

    def update(self, firing_states, topdown_indices=None):
        # firing_states: list of np.arrays, one per layer
        for scatter, state in zip(self.scatters, firing_states):
            colors = np.where(np.array(state) > 0, 'yellow', 'black')
            alphas = np.where(np.array(state) > 0, 0.9, 0.2)
            scatter.set_color(colors)
            scatter.set_alpha(alphas)
        # Optionally highlight top-down prediction flows
        if topdown_indices is not None:
            for i, indices in enumerate(topdown_indices):
                if indices is not None:
                    colors = np.array(self.scatters[i].get_facecolor())
                    colors[indices, :3] = [1, 0, 0]  # red for top-down
                    self.scatters[i].set_facecolor(colors)
        plt.pause(0.01)

    def animate(self, firing_states_seq, topdown_seq=None, interval=200):
        def anim_func(i):
            topdown = topdown_seq[i] if topdown_seq is not None else None
            self.update(firing_states_seq[i], topdown)
        ani = animation.FuncAnimation(self.fig, anim_func, frames=len(firing_states_seq), interval=interval, repeat=False)
        plt.show()
