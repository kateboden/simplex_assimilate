import numpy as np
import scipy
from matplotlib import pyplot as plt
from .shared_classes import Sample, Ensemble, ClassDirichlet, MixedDirichlet, ClassEnsemble

class Visualisation:

    def __init__(self, h_bnd, ax=None):
        self.h_bnd = h_bnd
        if not ax:
            fig, ax = plt.subplots(figsize=(20,8))
        self.width = 0.25
        self.bottom=-15
        ax.set_xlabel('Height')
        ax.set_ylabel('Log mass')
        self.ax = ax
        self.ax.set_ylim(self.bottom, 0)
        self.ax.set_xlim(-1, h_bnd[-1]+1)
        self.show_bounds()

    def show_bounds(self):
        for bound in self.h_bnd:
            self.ax.axvline(x=bound, color='r')
        self.ax.set_xticks(self.h_bnd)

    @staticmethod
    def random_color():
        return tuple(scipy.stats.uniform(0,1).rvs(3))

    def show_sample(self, sample: Sample):
        self.add_sample(sample, style='bar', show_labels=True, show_raw=True)

    def show_class_dirichlet(self, cd: ClassDirichlet, color=None):
        if not color:
            color = self.random_color()
        self.add_sample(cd.mean_sample, style='bar', show_labels=True, show_raw=True, color=color)
        self.ax.set_title(f'$\\alpha = {cd.alpha.sum():.2f}$')

    def show_dirichlet_bars(self, cd: ClassDirichlet, color=None, legend=None):
        if not color:
            color = self.random_color()
        x_r = self.h_bnd[cd.full_alpha[::2] > 0]  - self.width/2
        x_l = self.h_bnd[:-1][cd.full_alpha[1::2]> 0] + self.width/2
        x = np.sort(np.concatenate((x_r, x_l)))
        y_lims = np.maximum(np.log(cd.errorbars()), self.bottom)
        u = cd.alpha / cd.alpha.sum()
        y_error = np.maximum( (y_lims-np.log(u))*np.array([[-1.],[1.]]), 0)
        # print(y_lims, np.log(u), y_error)
        self.ax.errorbar(x=x, y=np.log(u), yerr=y_error, fmt='x', capsize=8, zorder=1, color=color, label=legend)
        self.ax.legend()
        self.ax.set_ylim(self.bottom, 0)

    def show_mixed_dirichlet(self, md: MixedDirichlet):
        for cd in md.dirichlets:
            self.show_dirichlet_bars(cd)


    def show_dirichlet_plus_samples(self, ce: ClassEnsemble, cd: ClassDirichlet, legend=None):
        color = self.random_color()
        self.show_class_ensemble(ce, color)
        self.show_dirichlet_bars(cd, color, legend=legend)

    def show_mixed_dirichlet_plus_samples(self, en: Ensemble, md: MixedDirichlet):
        for ce, cd, mr in zip(en.class_ensembles, md.dirichlets, md.mixing_rates):
            legend = f'$n = {len(ce.samples)}, \\pi = {mr:.2f}$'
            self.show_dirichlet_plus_samples(ce, cd, legend=legend)

    def show_class_ensemble(self, ce: ClassEnsemble, color=None):
        if not color:
            color = self.random_color()
        for sample in ce.samples:
            self.add_sample(sample, style='scatter', color=color, dot_size=50)

    def show_ensemble(self, ensemble: Ensemble):
        for ce in ensemble.class_ensembles:
            self.show_class_ensemble(ce)


    def add_sample(self, sample, style='bar', color='b', show_labels=False, show_raw=False, dot_size=10):
        assert len(sample) == 2*len(self.h_bnd) - 1

        x_r = self.h_bnd[sample[::2] > 0]  - self.width/2
        x_l = self.h_bnd[:-1][sample[1::2]> 0] + self.width/2
        top_r = np.log(sample[::2][sample[::2] > 0])
        top_l = np.log(sample[1::2][sample[1::2] > 0])
        if style=='bar':
            bars_r = self.ax.bar(x=x_r, height=top_r-self.bottom, width=self.width, color=color, bottom=self.bottom)
            bars_l = self.ax.bar(x=x_l, height=top_l-self.bottom, width=self.width, color=color, bottom=self.bottom)
            bars = [b for pair in zip(bars_r[:-1], bars_l) for b in pair] + [bars_r[-1],]
        elif style=='scatter':
            self.ax.scatter(x=x_r, y=top_r, color=color, s=dot_size)
            self.ax.scatter(x=x_l, y=top_l, color=color, s=dot_size )
        else:
            raise ValueError(f'Invalid style={style}, must be bar or scatter')

        pretty_float = lambda f: f'{f:.1e}' if 0<f<0.001 else f'{f:.3f}'

        if show_labels:
            assert style=='bar', 'labels can only be shown on bar charts. Try show_labels=False or style="bar"'
            labels = np.vectorize(pretty_float)(sample[sample>0])
            for bar, label in zip(bars, labels):
                height = bar.get_height()
                self.ax.annotate(
                    label,
                    xy=(bar.get_x() + bar.get_width() / 2, self.bottom + height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
                )

        if show_raw:
            for interval, l, r, in zip(self.h_bnd.intervals, sample[1::2], sample[2::2]):
                x_pos = sum(interval) / 2
                y_pos = self.bottom + 3
                a, v = l+r, interval[0]*l + interval[1]*r
                self.ax.annotate(
                    f'a={pretty_float(a)}\nv={pretty_float(v)}',
                    xy=(x_pos, y_pos),
                    ha='center',
                    va='bottom',
                    textcoords='offset points',
                    xytext=(0,5),
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
                )

        self.ax.set_ylim(self.bottom, 0)