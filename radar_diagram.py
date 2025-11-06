import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


class RadarDiagram:
    def radar_factory(self, num_vars, frame='circle'):
        theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

        class RadarAxes(PolarAxes):
            name = 'radar'
            RESOLUTION = 1

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.set_theta_zero_location('N')

            def fill(self, *args, closed=True, **kwargs):
                return super().fill(closed=closed, *args, **kwargs)

            def plot(self, *args, **kwargs):
                lines = super().plot(*args, **kwargs)
                for line in lines:
                    self._close_line(line)
                return lines

            def _close_line(self, line):
                x, y = line.get_data()
                if x[0] != x[-1]:
                    x = np.append(x, x[0])
                    y = np.append(y, y[0])
                    line.set_data(x, y)

            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(theta), labels)

            def _gen_axes_patch(self):
                if frame == 'circle':
                    return Circle((0.5, 0.5), 0.5)
                elif frame == 'polygon':
                    return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

            def _gen_axes_spines(self):
                if frame == 'circle':
                    return super()._gen_axes_spines()
                elif frame == 'polygon':
                    spine = Spine(axes=self, spine_type='circle',
                                  path=Path.unit_regular_polygon(num_vars))
                    spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                    return {'polar': spine}
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

        register_projection(RadarAxes)
        return theta

    def _render(self, data, label, title, restrictions, initial_data=None):
        N = 23
        theta = self.radar_factory(N, frame='polygon')

        fig, axs = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(top=0.85, bottom=0.05)

        axs.plot(theta, data, color='b', linewidth=2, label="Текущие характеристики")
        
        axs.plot(theta, restrictions, color='g', linestyle='--', linewidth=1.5, label="Предельные значения")
        
        if initial_data is not None:
            axs.plot(theta, initial_data, color='r', linestyle='-', linewidth=1.5, label="Начальные значения")

        axs.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize='small')
        axs.set_varlabels([f"X{i+1}" for i in range(N)])

        fig.text(0.5, 0.965, title, horizontalalignment='center', 
                 color='black', weight='bold', size='large')
        
        max_val = max(max(data), max(restrictions)) * 1.1
        axs.set_ylim(0, max_val)
        
        plt.draw()
        return fig

    def draw(self, filename, data, label, title, restrictions, initial_data=None):
        fig = self._render(data, label, title, restrictions, initial_data)
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    def draw_bytes(self, data, label, title, restrictions, initial_data=None):
        import io
        fig = self._render(data, label, title, restrictions, initial_data)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf.read()