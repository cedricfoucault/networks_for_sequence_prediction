import matplotlib as mpl
import matplotlib.pyplot as plt
import model
import numpy as np
import os
import training_data

STYLE_COMMON_RELATIVE_PATH = "styles/common.mplstyle"
STYLE_PAPER_RELATIVE_PATH = "styles/paper.mplstyle"
STYLE_PRESENTATION_RELATIVE_PATH = "styles/presentation.mplstyle"
STYLE_COMMON = os.path.join(os.path.dirname(os.path.realpath(__file__)), STYLE_COMMON_RELATIVE_PATH)
STYLE_PAPER = os.path.join(os.path.dirname(os.path.realpath(__file__)), STYLE_PAPER_RELATIVE_PATH)
STYLE_PRESENTATION = os.path.join(os.path.dirname(os.path.realpath(__file__)), STYLE_PRESENTATION_RELATIVE_PATH)

COLOR_LINE_NETWORK = "#ffa600"
COLOR_LINE_IO = "#7a5195"
COLOR_LINE_DELTARULE = "#ef5675"
COLOR_LINE_LEAKY = "#009999"
COLOR_OBSERVATION = "#000000"

COLOR_LINE_GRU = "#E69F00"
COLOR_LINE_ELMAN = "#F0E442"
COLOR_LINE_GRU_DIAGONAL = "#56B4E9"
COLOR_LINE_ELMAN_DIAGONAL = "#009E73"
COLOR_LINE_RESERVOIR = "#CC79A7"

COLOR_LINE_COUPLED = "#FF0000"
COLOR_LINE_INDEPENDENT = "#0000FF"

COLOR_BAR_ALTERNATIVE = (0.67, 0.67, 0.67)
COLOR_BAR_PERFORMANCE = "#6BA7EB"
COLOR_BAR_CONFIDENCE = "#ffa600"
COLOR_BAR_LOGODDS = "#1047A9"

COLOR_BAR_GRU = "#E69F00"
COLOR_BAR_ELMAN = "#F0E442"
COLOR_BAR_GRU_DIAGONAL = "#56B4E9"
COLOR_BAR_ELMAN_DIAGONAL = "#009E73"
COLOR_BAR_RESERVOIR = "#CC79A7"

MARKER_DEFAULT = ""
MARKER_ELMAN = "^"
MARKER_GRU_DIAGONAL = "s"
MARKER_RESERVOIR = "d"

COLOR_DOT_SAMPLE = (0, 0, 0, 0.3)

MARKERSIZE_OBSERVATION = 0.5
MARKERSIZE_PREDICTION = 1.5

BAR_HEIGHT = 0.8

ALPHA_CI = 0.2

AXISLABEL_PERFORMANCE = "% of optimal log likelihood"
AXISLABEL_PERFORMANCE_SHORT = "Performance (%)"
AXISLABEL_TIME = "Time step"
AXISLABEL_PREDICTION = "Predictions"
AXISLABEL_LEARNINGRATE = "Learning rate"
AXISLABEL_R2 = r"$R^2$ (% of variance explained)"

TEXTLABEL_CHANGEPOINT = "Change point"

def configure_plot_style(style="paper"):
    style_paths = [STYLE_COMMON]
    if style.lower() == "presentation":
        style_paths += [STYLE_PRESENTATION]
    else:
        style_paths += [STYLE_PAPER]
    plt.style.use(style_paths)

def get_formatter_percent():
    return plt.FuncFormatter("{:.0f}".format)

def get_formatter_percent_of_optimal(optimal_label="optimal"):
    format_fun = lambda value, tick_number: format_func_percent_of_optimal(value, tick_number, optimal_label)
    return plt.FuncFormatter(format_fun)

def get_formatter_strip_leading_0_with_n_decimals(n_decimals):
    return plt.FuncFormatter(get_format_func_strip_leading_0_with_n_decimals(n_decimals))

def get_format_func_strip_leading_0_with_n_decimals(n_decimals):
    return lambda value, tick_number: format_func_strip_leading_0(value, tick_number, n_decimals)

def format_func_strip_leading_0(value, tick_number, n_decimals=1):
    formatter = "{:" + ".{:d}f".format(n_decimals) + "}"
    if value == 0.:
        return "0"
    elif value == 1.:
        return "1"
    elif value == -1.:
        return "-1"
    elif value > 0 and value < 1:
        return formatter.format(value).lstrip('0')
    elif value < 0. and value > -1.:
        return "-" + formatter.format(-value).lstrip('0')
    else:
        return formatter.format(value)

def format_func_percent_of_optimal(value, tick_number, optimal_label):
    if value == 0:
        return "0\n(chance)"
    elif value == 100:
        return f"100\n({optimal_label})"
    else:
        return "{:.0f}".format(value)

def get_label_with_group_id(group_id, legend_style="default", detailed=False):
    model_i = training_data.load_models(group_id)[0]
    label = model_i.get_label(detailed=detailed)
    if isinstance(model_i, model.Network):
        if legend_style.lower() == "architecture":
            label = model_i.get_architecture_label(detailed=detailed)
            if model_i.has_train_outputonly():
                label = "Reservoir " + label
        elif legend_style.lower() == "mechanisms":
            label = model_i.get_mechanisms_label(detailed=detailed)
        elif legend_style.lower() == "gating":
            label = model_i.get_mechanisms_label(detailed=detailed, include_gating_type=True)
        elif legend_style.lower() == "training":
            label = model_i.get_mechanisms_label(detailed=detailed).replace('\n', ' ')

    return label

def get_sublabel_with_group_id(group_id, legend_style="default"):
    return None

def get_hidden_layer_label_with_group_id(group_id, legend_style="default"):
    if legend_style.lower() == "training":
        model_i = training_data.load_models(group_id)[0]
        if isinstance(model_i, model.Network) and model_i.has_train_outputonly():
            label = f"untrained\n hidden layer"
        else:
            gen_process_label = training_data.load_gen_process(group_id).get_label()
            label = f"trained on\n{gen_process_label}"
    else:
        label = get_label_with_group_id(group_id, legend_style=legend_style)
    return label

def get_line_color_with_group_id(group_id, legend_style="default"):
    model_i = training_data.load_models(group_id)[0]
    if isinstance(model_i, model.Network):
        unit_type_label = model_i.get_unit_type_label()
        is_elman = unit_type_label.lower() == "elman"
        has_train_outputonly = model_i.has_train_outputonly()
        has_diagonal_recurrent_weight_matrix = model_i.has_diagonal_recurrent_weight_matrix()
        if legend_style.lower() == "architecture":
            if has_train_outputonly:
                color = COLOR_LINE_RESERVOIR
            else:
                if is_elman:
                    if has_diagonal_recurrent_weight_matrix:
                        color = COLOR_LINE_ELMAN_DIAGONAL
                    else:
                        color = COLOR_LINE_ELMAN
                else:
                    if has_diagonal_recurrent_weight_matrix:
                        color = COLOR_LINE_GRU_DIAGONAL
                    else:
                        color = COLOR_LINE_GRU
        elif legend_style.lower() == "mechanisms" or legend_style.lower() == "gating":
            if has_train_outputonly or has_diagonal_recurrent_weight_matrix or is_elman:
                color = COLOR_BAR_ALTERNATIVE
            else:
                color = COLOR_LINE_NETWORK
        else:
            color = COLOR_LINE_NETWORK
    elif isinstance(model_i, model.DeltaRule):
        color = COLOR_LINE_DELTARULE
    elif isinstance(model_i, model.LeakyIntegrator):
        color = COLOR_LINE_LEAKY
    return color

def get_marker_with_group_id(group_id, legend_style="default"):
    model_i = training_data.load_models(group_id)[0]
    marker = MARKER_DEFAULT
    if isinstance(model_i, model.Network):
        unit_type_label = model_i.get_unit_type_label()
        is_elman = unit_type_label.lower() == "elman"
        has_train_outputonly = model_i.has_train_outputonly()
        has_diagonal_recurrent_weight_matrix = model_i.has_diagonal_recurrent_weight_matrix()
        if legend_style.lower() == "mechanisms" or legend_style.lower() == "gating":
            if has_train_outputonly:
                marker = MARKER_RESERVOIR
            else:
                if is_elman:
                    marker = MARKER_ELMAN
                else:
                    if has_diagonal_recurrent_weight_matrix:
                        marker = MARKER_GRU_DIAGONAL
                    else:
                        marker = MARKER_DEFAULT
    return marker

def get_bar_color_with_group_id(group_id, legend_style="default",
    test_gen_process=None, metric_kind="performance"):
    model_i = training_data.load_models(group_id)[0]
    training_gen_process = training_data.load_gen_process(group_id)
    return get_bar_color_with_model(model_i, training_gen_process, legend_style=legend_style, test_gen_process=test_gen_process, metric_kind=metric_kind)

def get_bar_color_with_model(model_i, training_gen_process, legend_style="default",
    test_gen_process=None, metric_kind="performance"):
    if metric_kind.lower() == "performance":
        color_match = COLOR_BAR_PERFORMANCE
    elif metric_kind.lower() == "confidence":
        color_match = COLOR_BAR_CONFIDENCE
    elif metric_kind.lower() == "logodds":
        color_match = COLOR_BAR_LOGODDS
    else:
        assert False, f"unknown metric_kind: {metric_kind}"

    if isinstance(model_i, model.Network):
        unit_type_label = model_i.get_unit_type_label()
        has_train_outputonly = model_i.has_train_outputonly()
        has_diagonal_recurrent_weight_matrix = model_i.has_diagonal_recurrent_weight_matrix()
        is_elman = unit_type_label.lower() == "elman"
        if legend_style.lower() == "architecture":
            if has_train_outputonly:
                color = COLOR_BAR_RESERVOIR
            else:
                if is_elman:
                    if has_diagonal_recurrent_weight_matrix:
                        color = COLOR_BAR_ELMAN_DIAGONAL
                    else:
                        color = COLOR_BAR_ELMAN
                else:
                    if has_diagonal_recurrent_weight_matrix:
                        color = COLOR_BAR_GRU_DIAGONAL
                    else:
                        color = COLOR_BAR_GRU
        elif (legend_style.lower() == "mechanisms"
            or legend_style.lower() == "gating"
            or legend_style.lower() == "training"):
            if has_train_outputonly or has_diagonal_recurrent_weight_matrix or is_elman:
                color = COLOR_BAR_ALTERNATIVE
            else:
                color = color_match
        else:
            color = color_match
    else:
        color = COLOR_BAR_ALTERNATIVE
    return color


def configure_bar_plot_axes(ax, show_minor=False, labelbottom=True):
    # Provide tick lines along the xticks
    ax.grid(True, axis='x', which='major', ls='--', lw=.5, c='k', alpha=.3)
    if show_minor:
        ax.grid(True, axis='x', which='minor', ls='--', lw=.5, c='k', alpha=.3)
    # Remove the tick marks; they are unnecessary with the tick
    ax.tick_params(axis='both', which='both',
                   bottom=False, top=False, labelbottom=labelbottom,
                   left=False, right=False, labelleft=True)
    # Remove the plot frame lines. They are unnecessary here.
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

#### OUTDATED

def get_formatter_r2_with_large_effect():
    return plt.FuncFormatter(format_func_r2_with_large_effect)

def format_func_r2_with_large_effect(value, tick_number):
    if value == 25:
        return "25%\nlarge"
    else:
        return "{:.0f}%".format(value)

def plot_sequence(figid, labels, sequences, markers, colors, markersize=3.0, linewidth=1.5, show=True, vlines=[], title="Sample sequence", ylabel=None, ylim=None):
    figsize = (7.2, 3.6)
    plt.close(figid)
    fig = plt.figure(figid, figsize)
    plt.title(title)
    for xc in vlines:
        plt.axvline(x=xc, color="#000000", linewidth=1.0, linestyle="--")
    for i, label in enumerate(labels):
        plt.plot(sequences[i], markers[i], label = label, color = colors[i], markersize=markersize, linewidth=linewidth)
    plt.xlabel("time step")
    plt.legend()
    if ylabel != None:
        plt.ylabel(ylabel)
    if ylim != None:
        plt.ylim(ylim)
    if show:
        plt.show()
    return fig

def plot_simulation_sequence_bernoulli(figid, p_gen, input, net_estimate, io_estimate, with_deltarule=False, dr_estimate=None, ylim_estimate=None):
    # Vertical line  for change points
    vlines = [t for t in range(len(p_gen) - 1) if p_gen[t + 1] != p_gen[t]]
    # Subplot: estimates
    labels1 = []
    sequences1 = []
    markers1 = []
    colors1 = []
    # generative probability sequence
    labels1.append("Generative process")
    sequences1.append(p_gen)
    markers1.append("-")
    colors1.append("#3b15f8")
    # input sequence
    labels1.append("Observation")
    sequences1.append(input)
    markers1.append("x")
    colors1.append("#7f7f7f")
    # RNN estimate sequence
    labels1.append("RNN")
    markers1.append(".")
    sequences1.append(net_estimate)
    colors1.append(COLOR_LINE_NETWORK)
    # IO estimate sequence
    labels1.append("Ideal observer")
    sequences1.append(io_estimate)
    markers1.append(".")
    colors1.append(COLOR_LINE_IO)
    if with_deltarule:
        # Delta Rule estimate sequence
        labels1.append("Delta rule")
        sequences1.append(dr_estimate)
        markers1.append(".")
        colors1.append(COLOR_LINE_DELTARULE)
    return plot_sequence(figid, vlines=vlines,
    sequences=sequences1, labels=labels1, markers=markers1, colors=colors1, markersize=2.0, linewidth=1.0, ylabel="Probability", ylim=ylim_estimate)

def plot_simulation_sequence_markov(figid, p_gen_0, p_gen_1, input, net_estimate, io_estimate, with_deltarule=False, dr_estimate=None, title="Sample Sequence"):
    # Vertical line  for change points
    vlines = [t for t in range(len(p_gen_0) - 1) if p_gen_0[t + 1] != p_gen_0[t]]
    # Subplot: estimates
    labels1 = []
    sequences1 = []
    markers1 = []
    colors1 = []
    labels2 = []
    sequences2 = []
    markers2 = []
    colors2 = []
    # generative probability sequence
    labels1.append("Generative process" + " | 0")
    sequences1.append(p_gen_0)
    markers1.append("-")
    colors1.append("#f81592")
    # | 1
    labels1.append("Generative process" + " | 1")
    sequences1.append(p_gen_1)
    markers1.append("-")
    colors1.append("#2fa9ea")
    # input sequence
    labels1.append("Observation")
    sequences1.append(input)
    markers1.append("x")
    colors1.append("#7f7f7f")
    # RNN estimate sequence
    labels1.append("RNN")
    markers1.append(".")
    sequences1.append(net_estimate)
    colors1.append(COLOR_LINE_NETWORK)
    # IO estimate sequence
    labels1.append("Ideal observer")
    sequences1.append(io_estimate)
    markers1.append(".")
    colors1.append(COLOR_LINE_IO)
    if with_deltarule:
        labels1.append("Delta rule")
        sequences1.append(dr_estimate)
        markers1.append(".")
        colors1.append(COLOR_LINE_DELTARULE)
    figsize = (9.6, 6.4)
    plt.close(figid)
    fig = plt.figure(figid, figsize)
    plt.title(title)
    draw_sequence(labels1, sequences1, markers1, colors1, ylabel="Probability", ylim=[0., 1.], vlines=vlines, markersize=3.0, linewidth=1.5)
    plt.xlabel("time step")
    return fig

def draw_sequence(labels, sequences, markers, colors, ylabel=None, ylim=None, vlines=[], markersize=3.0, linewidth=1.5):
    for xc in vlines:
        plt.axvline(x=xc, color="#000000", linewidth=1.0, linestyle="--")
    for i, label in enumerate(labels):
        plt.plot(sequences[i], markers[i], label = label, color = colors[i], markersize=markersize, linewidth=linewidth)
    if ylabel != None:
        plt.ylabel(ylabel)
    if ylim != None:
        plt.ylim(ylim)
    plt.legend()
