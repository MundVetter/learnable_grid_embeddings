from utils import get_1d_sincos_pos_embed_from_grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class FourierVisualizer:
    def __init__(self, fig, ax, sequence_length = 16, embed_dim = 14, factor = 10_000):
        self.fig = fig
        self.ax = ax
        self.lines = []

        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.factor = factor

    def update_sequence_length(self, val):
        self.sequence_length = int(val)
        self.ax.cla()
        self.lines = []
        self.update()

    def update_embed_dim(self, val):
        self.embed_dim = int(val)
        self.ax.cla()
        self.lines = []
        self.update()
    
    def update_factor(self, val):
        self.factor = val
        self.update()

    def update(self):
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.embed_dim, np.arange(self.sequence_length), self.factor)
        self.plot_embed(pos_embed)
        self.fig.canvas.draw_idle()

    def plot_embed(self, pos_embed):
        embed_dim = pos_embed.shape[1]
        lines = []
        for i in range(embed_dim):
            if len(self.lines) == 0:
                line, = self.ax.plot(pos_embed[:, i], label=f"dim {i}")
                lines.append(line)
            else:
                self.lines[i].set_ydata(pos_embed[:, i])

        if len(self.lines) == 0:
            self.lines = lines



if __name__ == "__main__":
    sequence_length = 16
    embed_dim = 14
    factor = 10_000

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    vis = FourierVisualizer(fig, ax, sequence_length, embed_dim, factor)
    vis.update()
    ax.set_xlabel('Input')
    ax.set_xlabel('Output')

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(bottom=0.25)

    # Make a horizontal slider to control the sequence length.
    axseq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    seq_slider = Slider(
        ax=axseq,
        label='Sequence length',
        valmin=1,
        valmax=300,
        valinit=sequence_length,
        valstep=1
    )

    # make a horizontal slider to control embed dim
    axdim = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    dim_slider = Slider(
        ax=axdim,
        label='Embed dim',
        valmin=2,
        valmax=300,
        valinit=embed_dim,
        valstep=2
    )

    # make a horizontal slider to control factor
    axfac = fig.add_axes([0.25, 0.2, 0.65, 0.03])
    fac_slider = Slider(
        ax=axfac,
        label='Factor',
        valmin=1,
        valmax=10_000,
        valinit=factor,
        valstep=1
    )

    # make a factor button with smaller max_factor
    axfacsmall = fig.add_axes([0.25, 0.25, 0.65, 0.04])
    fac2_slider = Slider(
        ax=axfacsmall,
        label='factor precise',
        valmin=1,
        valmax=100,
        valinit=factor,
        valstep=1
    )

    fac2_slider.on_changed(vis.update_factor)
    seq_slider.on_changed(vis.update_sequence_length)
    dim_slider.on_changed(vis.update_embed_dim)
    fac_slider.on_changed(vis.update_factor)

    # Add a button for resetting the parameters
    # resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    # button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    plt.show()




    # button.on_clicked(reset)
