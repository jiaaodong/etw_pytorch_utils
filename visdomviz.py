import numpy as np
import visdom


class VisdomViz(object):

    def __init__(
            self, env_name='main', *, server='http://localhost', port=8097
    ):
        print('=====>')
        print('Initializing vizdom env [{}]'.format(env_name))
        print('server: {}, port: {}'.format(server, port))

        self.viz = visdom.Visdom(server=server, port=port, env=env_name)
        self.wins = {}
        self.update_callbacks = {}
        print('<=====')

    def text(self, _text, win=None):
        self.viz.text(_text, win=win)

    def update(self, mode, it, eval_dict):
        for k, v in eval_dict.items():
            if k in self.update_callbacks:
                self.update_callbacks[k](self, mode, it, k, v)

    def add_callback(self, name, cb):
        self.update_callbacks[name] = cb

    def add_callbacks(self, cbs={}, **kwargs):
        cbs = {**cbs, **kwargs}
        for name, cb in cbs.items():
            self.add_callback(name, cb)

    def append_element(self, window_name, x, y, line_name, xlabel='iterations'):
        r"""
            Appends an element to a line

        Paramters
        ---------
        key: str
            Name of window
        x: float
            x-value
        y: float
            y-value
        line_name: str
            Name of line
        xlabel: str
        """

        if window_name in self.wins:
            self.viz.updateTrace(
                X=np.array([x]),
                Y=np.array([y]),
                win=self.wins[window_name],
                name=line_name
            )
        else:
            self.wins[window_name] = self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                opts=dict(
                    xlabel=xlabel,
                    ylabel=window_name,
                    title=window_name,
                    marginleft=30,
                    marginright=30,
                    marginbottom=30,
                    margintop=30,
                    legend=[line_name]
                )
            )
