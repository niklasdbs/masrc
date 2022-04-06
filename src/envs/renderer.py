from collections import defaultdict
from datetime import datetime
from numbers import Number

import cv2
import networkx as nx
import numpy as np
import pylab as pl
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from envs.agent_state import AgentState
from envs.enums import ParkingStatus

COLOR_MAP = {
        ParkingStatus.FREE: "green",
        ParkingStatus.IN_VIOLATION: "red",
        ParkingStatus.FINED: "yellow",
        ParkingStatus.OCCUPIED: "orange"
}


class Renderer:

    def __init__(self, config):
        self._canvas = None  # used to cache canvas that is related to drawing
        self.render_width = config.render_resolution.w
        self.render_height = config.render_resolution.h
        self._render_dpi = config.render_resolution.dpi
        self._rendered_steps = []
        self._fig = None  # cache figure for rendering
        self.plot_q_values_as_heatmap = True  # todo do not hardcode
        self._q_values = defaultdict(lambda: None)

    def add_q_values(self, agent, qs):
        self._q_values[agent] = qs

    def _fast_render(self, fig, ax, resources_draw, agents_draw, other_draw):
        if not self._canvas:
            self._canvas = FigureCanvasAgg(fig)

            self._canvas.draw()  # update/draw the elements
            self.background = self._canvas.copy_from_bbox(fig.bbox)
        else:
            self._canvas.restore_region(self.background)

        for r in resources_draw:
            ax.draw_artist(r)

        for o in other_draw:
            ax.draw_artist(o)

        for a in agents_draw:
            ax.draw_artist(a)

        self._canvas.blit(fig.bbox)

        # get the width and the height to resize the matrix
        l, b, w, h = self._canvas.figure.bbox.bounds
        w, h = int(w), int(h)

        #  exports the canvas to a string buffer and then to a numpy nd.array
        return cv2.cvtColor(np.array(self._canvas.renderer._renderer), cv2.COLOR_RGB2BGR)

        # buf = self._canvas.tostring_rgb()
        # image = np.frombuffer(buf, dtype=np.uint8)
        # return image
        # return image.reshape(3, h, w)

    def render(self, env, mode, show, additional_info=None):
        # https://stackoverflow.com/questions/8955869/why-is-plotting-with-matplotlib-so-slow/8956211#8956211
        if not self._fig:
            self._fig, self._ax = plt.subplots(figsize=(
                    self.render_width / self._render_dpi, self.render_height / self._render_dpi),
                    dpi=self._render_dpi)  # cache figure and axex, as we resuse them
            self._fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            y = nx.get_node_attributes(env.graph, "y")
            x = nx.get_node_attributes(env.graph, "x")
            self._ax.axis("off")
            self._ax.set_xlim(np.min(list(x.values())), np.max(list(x.values())))
            self._ax.set_ylim(np.min(list(y.values())), np.max(list(y.values())))
            self._ax.get_xaxis().set_visible(False)
            self._ax.get_yaxis().set_visible(False)
            # nx.rescale_layout_dict(pos, 2)
            self._pos_for_drawing = {key: (x[key], y[key]) for key in
                                     y.keys() & x.keys()}  # cache these for performance

            # nx.draw(env.graph, pos=self._pos_for_drawing, ax=self._ax, node_size=0, arrows=False, with_labels=False)
            nx.draw_networkx_nodes(env.graph, pos=self._pos_for_drawing, ax=self._ax, node_size=0)

            edge_to_int = {e: i for i, e in enumerate(env.graph.edges)}
            self._edge_cmap = pl.get_cmap("copper")
            self._edge_colors = np.zeros(shape=len(edge_to_int))
            edge_widths = np.full(shape=len(edge_to_int), fill_value=0.5)


            self._edge_id_to_graph_int = {}
            for e, i in env.edge_to_edge_id_mapping.items():
                self._edge_id_to_graph_int[i] = edge_to_int[e]
                edge_widths[edge_to_int[e]] = 2.0

            self._edge_collection = draw_networkx_edges(env.graph,
                                                        pos=self._pos_for_drawing,
                                                        ax=self._ax,
                                                        arrows=False,
                                                        width=edge_widths,
                                                        edge_vmin=0,
                                                        edge_vmax=1,
                                                        edge_cmap=self._edge_cmap,
                                                        edge_color=self._edge_colors,
                                                        )

            self._resource_drawing_cache = [
                    self._ax.plot(resource.x,
                                  resource.y,
                                  linestyle='',
                                  marker='o',
                                  markersize=5,
                                  markeredgecolor='none',
                                  color=COLOR_MAP[resource.status],
                                  zorder=-5,
                                  animated=True)[0] for resource in
                    env.resources]
            self._agent_drawing_cache = [self._ax.scatter(0, 0, s=0, c="cyan", zorder=20, animated=True) for _ in #todo 75
                                         env.agent_states]

            self._agent_target_drawing_cache = [self._ax.scatter(0, 0, s=0, c="purple", zorder=10, animated=True) #todo
                                                for _ in env.agent_states]

        else:
            for resource_drawing, resource in zip(self._resource_drawing_cache, env.resources):
                resource_drawing.set_color(COLOR_MAP[resource.status])

        for agent_target_plot, agent_plot, agent_state in zip(self._agent_target_drawing_cache,
                                                              self._agent_drawing_cache,
                                                              env.agent_states):
            agent_state: AgentState
            agent_position_source = agent_state.position_node_source
            agent_position_node = agent_state.position_node

            x, y = self._pos_for_drawing[agent_position_node]
            x_source, y_source = self._pos_for_drawing[agent_position_source]

            ratio = agent_state.position_on_edge / env.graph[agent_position_source][agent_position_node]["length"]

            x_on_edge = x_source + ((x - x_source) * ratio)
            y_on_edge = y_source + ((y - y_source) * ratio)

            agent_plot.set_offsets([x_on_edge, y_on_edge])

            if agent_state.current_route is not None:
                x, y = self._pos_for_drawing[agent_state.current_route[-1]]
                agent_target_plot.set_offsets([x, y])
            else:
                agent_target_plot.set_offsets([0.0, 0.0])

        if self._q_values[0] is not None:
            qs = self._q_values[0]  # todo do not hardcode which q values of agents to plot
            qs = (qs - qs.min()) / (qs.max() - qs.min())

            for i, q in enumerate(qs[0][0]):
                self._edge_colors[self._edge_id_to_graph_int[i]] = q

            self._edge_collection.set_array(self._edge_colors)

        current_time = datetime.fromtimestamp(env.current_time)
        info_text = self._ax.text(0.0, 0.0,
                                  f"{env.current_day}:{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}. "
                                  f"#cur v: {sum(1 for r in env.resources if r.status == ParkingStatus.IN_VIOLATION)} "
                                  f"#f: {env.fined_resources}"
                                  f"sum v: {env.cumulative_resources_in_violation}",
                                  fontsize=40,
                                  horizontalalignment='left',
                                  verticalalignment='bottom',
                                  transform=self._ax.transAxes,
                                  animated=True)

        img = self._fast_render(self._fig,
                                self._ax,
                                self._resource_drawing_cache,
                                self._agent_drawing_cache,
                                [self._edge_collection]  + self._agent_target_drawing_cache)#todo + [info_text]

        if show:
            plt.show()

        if mode == "internal_step":
            self._rendered_steps.append(img)
            return [img]
        else:
            ret = self._rendered_steps
            self._rendered_steps = []
            return ret


def draw_networkx_edges(
        G,
        pos,
        edgelist=None,
        width=1.0,
        edge_color="k",
        style="solid",
        alpha=None,
        arrowstyle="-|>",
        arrowsize=10,
        edge_cmap=None,
        edge_vmin=None,
        edge_vmax=None,
        ax=None,
        arrows=True,
        label=None,
        node_size=300,
        nodelist=None,
        node_shape="o",
        connectionstyle=None,
        min_source_margin=0,
        min_target_margin=0,
):
    """Draw the edges of the graph G.

    This draws only the edges of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    edgelist : collection of edge tuples
       Draw only specified edges(default=G.edges())

    width : float, or array of floats
       Line width of edges (default=1.0)

    edge_color : color or array of colors (default='k')
       Edge color. Can be a single color or a sequence of colors with the same
       length as edgelist. Color can be string, or rgb (or rgba) tuple of
       floats from 0-1. If numeric values are specified they will be
       mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    style : string
       Edge line style (default='solid') (solid|dashed|dotted,dashdot)

    alpha : float
       The edge transparency (default=None)

    edge_ cmap : Matplotlib colormap
       Colormap for mapping intensities of edges (default=None)

    edge_vmin,edge_vmax : floats
       Minimum and maximum for edge colormap scaling (default=None)

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    arrows : bool, optional (default=True)
       For directed graphs, if True draw arrowheads.
       Note: Arrows will be the same color as edges.

    arrowstyle : str, optional (default='-|>')
       For directed graphs, choose the style of the arrow heads.
       See :py:class: `matplotlib.patches.ArrowStyle` for more
       options.

    arrowsize : int, optional (default=10)
       For directed graphs, choose the size of the arrow head head's length and
       width. See :py:class: `matplotlib.patches.FancyArrowPatch` for attribute
       `mutation_scale` for more info.

    connectionstyle : str, optional (default=None)
       Pass the connectionstyle parameter to create curved arc of rounding
       radius rad. For example, connectionstyle='arc3,rad=0.2'.
       See :py:class: `matplotlib.patches.ConnectionStyle` and
       :py:class: `matplotlib.patches.FancyArrowPatch` for more info.

    label : [None| string]
       Label for legend

    min_source_margin : int, optional (default=0)
       The minimum margin (gap) at the begining of the edge at the source.

    min_target_margin : int, optional (default=0)
       The minimum margin (gap) at the end of the edge at the target.

    Returns
    -------
    matplotlib.collection.LineCollection
        `LineCollection` of the edges

    list of matplotlib.patches.FancyArrowPatch
        `FancyArrowPatch` instances of the directed edges

    Depending whether the drawing includes arrows or not.

    Notes
    -----
    For directed graphs, arrows are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False. Be sure to include `node_size` as a
    keyword argument; arrows are drawn considering the size of nodes.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))

    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 3)])
    >>> arcs = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    >>> alphas = [0.3, 0.4, 0.5]
    >>> for i, arc in enumerate(arcs):  # change alpha values of arcs
    ...     arc.set_alpha(alphas[i])

    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_labels()
    draw_networkx_edge_labels()
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import colorConverter, Colormap, Normalize
        from matplotlib.collections import LineCollection
        from matplotlib.patches import FancyArrowPatch
        import numpy as np
    except ImportError as e:
        raise ImportError("Matplotlib required for draw()") from e
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list(G.edges())

    if len(edgelist) == 0:  # no edges!
        if not G.is_directed() or not arrows:
            return LineCollection(None)
        else:
            return []

    if nodelist is None:
        nodelist = list(G.nodes())

    # FancyArrowPatch handles color=None different from LineCollection
    if edge_color is None:
        edge_color = "k"

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    # Check if edge_color is an array of floats and map to edge_cmap.
    # This is the only case handled differently from matplotlib
    # if (
    #     np.iterable(edge_color)
    #     and (len(edge_color) == len(edge_pos))
    #     and np.alltrue([isinstance(c, Number) for c in edge_color])
    # ):
    #     if edge_cmap is not None:
    #         assert isinstance(edge_cmap, Colormap)
    #     else:
    #         edge_cmap = plt.get_cmap()
    #     if edge_vmin is None:
    #         edge_vmin = min(edge_color)
    #     if edge_vmax is None:
    #         edge_vmax = max(edge_color)
    #     color_normal = Normalize(vmin=edge_vmin, vmax=edge_vmax)
    #     edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    if not G.is_directed() or not arrows:
        edge_collection = LineCollection(
                edge_pos,
                array=edge_color,
                linewidths=width,
                antialiaseds=False,
                linestyle=style,
                transOffset=ax.transData,
                alpha=alpha,
                animated=True
        )

        edge_collection.set_cmap(edge_cmap)
        edge_collection.set_clim(edge_vmin, edge_vmax)

        edge_collection.set_zorder(1)  # edges go behind nodes
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)

        return edge_collection
