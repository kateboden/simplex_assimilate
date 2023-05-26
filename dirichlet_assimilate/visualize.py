import numpy as np
import scipy
from matplotlib import pyplot as plt
from .shared_classes import Sample, Ensemble, ClassDirichlet, MixedDirichlet, ClassEnsemble, RawSample, RawEnsemble, Observation, HeightBounds

class Visualisation:

    def __init__(self, h_bnd: HeightBounds, ax=None, log_scale=True, figsize=(20,8)):
        self.h_bnd = h_bnd
        self.log_scale = log_scale
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        self.width = 0.25
        self.ax = ax
        self.ax.set_xlabel('Height')
        self.ax.set_xlim(-1, h_bnd[-1]+1)
        self.show_bounds()
        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'].__iter__()

        if self.log_scale:
            self.ax.set_ylabel('Log area')
            self.bottom = -15
            self.ax.set_ylim(self.bottom, 0)
        else:
            # set regular scale
            self.ax.set_ylim(0,1)
            self.ax.set_ylabel('Area')
            self.bottom = 0

    def show_bounds(self):
        for bound in self.h_bnd:
            self.ax.axvline(x=bound, color='r')
        self.ax.set_xticks(self.h_bnd)

    @staticmethod
    def pretty_float(f):
        return f'{f:.1e}' if 0<f<0.001 else f'{f:.3f}'

    def show_sample(self, sample: Sample):
        self.add_sample(sample, style='bar', show_labels=True, show_raw=True)

    def show_raw_sample(self, raw_sample: RawSample):
        self.add_raw_sample(raw_sample, style='bar', show_labels=True)

    def show_raw_ensemble(self, raw_ensemble: RawEnsemble):
        for rs in raw_ensemble.samples:
            self.add_raw_sample(rs, style='scatter', dot_size=50)

    def show_class_dirichlet(self, cd: ClassDirichlet, color=None):
        if not color:
            color = next(self.color_cycle)
        self.add_sample(cd.mean_sample, style='bar', show_labels=True, show_raw=True, color=color)
        self.ax.set_title(f'$\\alpha = {cd.alpha.sum():.2f}$')

    def show_dirichlet_bars(self, cd: ClassDirichlet, color=None, legend=None):
        if not color:
            color = next(self.color_cycle)
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
        color = next(self.color_cycle)
        self.show_class_ensemble(ce, color)
        self.show_dirichlet_bars(cd, color, legend=legend)

    def show_mixed_dirichlet_plus_samples(self, en: Ensemble, md: MixedDirichlet):
        for ce, cd, mr in zip(en.class_ensembles, md.dirichlets, md.mixing_rates):
            legend = f'$n = {len(ce.samples)}, \\pi = {mr:.2f}$'
            self.show_dirichlet_plus_samples(ce, cd, legend=legend)

    def show_class_ensemble(self, ce: ClassEnsemble, color=None):
        if not color:
            color = next(self.color_cycle)
        for sample in ce.samples:
            self.add_sample(sample, style='scatter', color=color, dot_size=50)

    def show_ensemble(self, ensemble: Ensemble):
        for ce in ensemble.class_ensembles:
            self.show_class_ensemble(ce)

    def add_raw_sample(self, rs: RawSample, style='bar', show_labels=False, color='b', dot_size=10):
        x = [np.mean(interval) for interval in self.h_bnd.intervals]
        x.insert(0, 0.)
        top = np.insert(rs.area, 0, 1-sum(rs.area))
        if self.log_scale:
            top = np.log(top)


        if style=='bar':
            bars = self.ax.bar(x=x, height=top-self.bottom, width=self.width, color=color, bottom=self.bottom)
        elif style=='scatter':
            self.ax.scatter(x=x, y=top, color=color, s=dot_size)
        else:
            raise ValueError(f'Invalid style={style}, must be bar or scatter')

        if show_labels:
            assert style=='bar', 'labels can only be shown on bar charts. Try show_labels=False or style="bar"'
            for bar, t in zip(bars, top):
                # height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width() / 2, self.bottom + bar.get_height(), self.pretty_float(t), ha='center', va='bottom')

    def add_sample(self, sample, style='bar', color='b', show_labels=False, show_raw=False, dot_size=10):
        assert len(sample) == 2*len(self.h_bnd) - 1

        x_r = self.h_bnd[sample[::2] > 0]  - self.width/2
        x_l = self.h_bnd[:-1][sample[1::2]> 0] + self.width/2
        top_r = sample[::2][sample[::2] > 0]
        top_l = sample[1::2][sample[1::2] > 0]

        if self.log_scale:
            top_r, top_l = np.log(top_r), np.log(top_l)

        if style=='bar':
            bars_r = self.ax.bar(x=x_r, height=top_r-self.bottom, width=self.width, color=color, bottom=self.bottom)
            bars_l = self.ax.bar(x=x_l, height=top_l-self.bottom, width=self.width, color=color, bottom=self.bottom)
            bars = [b for pair in zip(bars_r[:-1], bars_l) for b in pair] + [bars_r[-1],]
        elif style=='scatter':
            self.ax.scatter(x=x_r, y=top_r, color=color, s=dot_size)
            self.ax.scatter(x=x_l, y=top_l, color=color, s=dot_size )
        else:
            raise ValueError(f'Invalid style={style}, must be bar or scatter')


        if show_labels:
            assert style=='bar', 'labels can only be shown on bar charts. Try show_labels=False or style="bar"'
            labels = np.vectorize(self.pretty_float)(sample[sample>0])
            for bar, label in zip(bars, labels):
                # height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width() / 2, self.bottom + bar.get_height(), label, ha='center', va='bottom')

        if show_raw:
            for interval, l, r, in zip(self.h_bnd.intervals, sample[1::2], sample[2::2]):
                x_pos = sum(interval) / 2
                y_pos = self.bottom + 3 if self.log_scale else 0.6
                a, v = l+r, interval[0]*l + interval[1]*r
                self.ax.annotate(
                    f'a={self.pretty_float(a)}\nv={self.pretty_float(v)}',
                    xy=(x_pos, y_pos),
                    ha='center',
                    va='bottom',
                    textcoords='offset points',
                    xytext=(0,5),
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
                )
