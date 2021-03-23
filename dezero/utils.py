import os
import subprocess


def _dot_var(v, verbose=False):
    dot_var = '{id} [label="{label}", color=orange, style=filled]\n'

    name = "" if v.name is None else v.name

    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)
    return dot_var.format(id=id(v), label=name)


def _dot_func(f):
    dot_func = '{id} [label="{label}", color=lightblue, ' \
               'style=filled, shape=box]\n'
    txt = dot_func.format(id=id(f), label=f.__class__.__name__)

    dot_edge = '{start} -> {end}\n'
    for x in f.inputs:
        txt += dot_edge.format(start=id(x), end=id(f))
    for y in f.outputs:
        txt += dot_edge.format(start=id(f), end=id(y()))
    return txt


def get_dot_graph(output, verbose=True):
    txt = ""
    funcs = list()
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose=verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose=verbose)

            if x.creator is not None:
                add_func(x.creator)
    return "digraph g {\n" + txt + "}"


def plot_dot_graph(output, verbose=True, to_file="graph.png"):
    dot_graph = get_dot_graph(output, verbose=verbose)

    tmp_dir = os.path.join(os.path.expanduser("~"), ".dezero")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(graph_path, "w") as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    cmd = "dot {} -T {} -o {}".format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)
