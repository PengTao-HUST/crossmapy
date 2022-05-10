# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from sklearn import metrics

from .utils import two_array_to_edges_set


def plot_ratios_curve(ratios, label=None, diag_line=True, grid=True, tight_layout=True,
                      xlabel='Normalized $r$', ylabel='Normalized $IC$', **kwargs):
    n_ele = len(ratios)
    neighbor_ratios = np.arange(n_ele) / (n_ele - 1)

    if label is None:
        auc = metrics.auc(neighbor_ratios, ratios)
        label = f'AUC({auc:.3f})'

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(neighbor_ratios, ratios, label=label, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='lower right')
    # ax.axis('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if diag_line:
        ax.plot([0, 1], [0, 1], ls='--', c='grey')

    if grid:
        ax.grid()

    if tight_layout:
        fig.tight_layout()

    return fig, ax


def plot_mul_ratios_curve(ratios_list, colors=None, labels=None, diag_line=True,
                          legend_loc=None, grid=True, title=None, tight_layout=True,
                          xlabel='Normalized $r$', ylabel='Normalized $IC$', **kwargs):
    n_curve = len(ratios_list)
    fig, ax = plt.subplots(figsize=(5, 5))

    for i in range(n_curve):
        ratios = ratios_list[i]
        n_ele = len(ratios)
        neighbor_ratios = np.arange(n_ele) / (n_ele - 1)
        auc = metrics.auc(neighbor_ratios, ratios)

        if labels is None:
            label = f'AUC{i + 1}({auc:.3f})'
        else:
            label = f'{labels[i]}({auc:.3f})'

        if colors is None:
            ax.plot(neighbor_ratios, ratios, label=label, **kwargs)
        else:
            ax.plot(neighbor_ratios, ratios, color=colors[i], label=label, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.axis('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if diag_line:
        ax.plot([0, 1], [0, 1], ls='--', c='grey')

    if title is not None:
        ax.set_title(title)

    if legend_loc is None:
        loc = 'lower right'
    else:
        loc = legend_loc
    ax.legend(loc=loc)

    if grid:
        ax.grid()

    if tight_layout:
        fig.tight_layout()

    return fig, ax


def plot_roc_curve(fpr, tpr, label=None, diag_line=True, grid=True, tight_layout=True, **kwargs):
    if label is None:
        roc_auc = metrics.auc(fpr, tpr)
        label = f'AUC({roc_auc:.3f})'

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=label, **kwargs)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    # ax.axis('equal')

    if grid:
        ax.grid()

    if diag_line:
        ax.plot([0, 1], [0, 1], ls='--', c='grey')

    if tight_layout:
        fig.tight_layout()

    return fig, ax


def plot_mul_roc_curve(fpr_list, tpr_list, colors=None, labels=None, diag_line=True,
                       legend_loc=None, grid=True, title=None, tight_layout=True,
                       xlabel='FPR', ylabel='TPR', **kwargs):
    n_curve = len(fpr_list)
    fig, ax = plt.subplots(figsize=(5, 5))

    for i in range(n_curve):
        fpr = fpr_list[i]
        tpr = tpr_list[i]
        roc_auc = metrics.auc(fpr, tpr)

        if labels is None:
            label = f'AUC{i + 1}({roc_auc:.3f})'
        else:
            label = f'{labels[i]}({roc_auc:.3f})'

        if colors is None:
            ax.plot(fpr, tpr, label=label, **kwargs)
        else:
            ax.plot(fpr, tpr, color=colors[i], label=label, **kwargs)

    if diag_line:
        ax.plot([0, 1], [0, 1], ls='--', c='grey')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.axis('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if title is not None:
        ax.set_title(title)

    if legend_loc is None:
        loc = 'lower right'
    else:
        loc = legend_loc
    ax.legend(loc=loc)

    if grid:
        ax.grid()

    if tight_layout:
        fig.tight_layout()

    return fig, ax


def plot_pr_curve(precision, recall, label=None, diag_line=True, grid=True, tight_layout=True, **kwargs):
    if label is None:
        roc_auc = metrics.auc(recall, precision)
        label = f'AUC({roc_auc:.3f})'

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=label, **kwargs)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower right')
    # ax.axis('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if diag_line:
        ax.plot([0, 1], [1, 0], ls='--', c='grey')

    if grid:
        ax.grid()

    if tight_layout:
        fig.tight_layout()

    return fig, ax


def plot_mul_pr_curve(precision_list, recall_list, colors=None, labels=None, diag_line=True,
                      legend_loc=None, grid=True, title=None, tight_layout=True, **kwargs):
    n_curve = len(precision_list)
    fig, ax = plt.subplots(figsize=(5, 5))

    for i in range(n_curve):
        precision = precision_list[i]
        recall = recall_list[i]
        roc_auc = metrics.auc(recall, precision)

        if labels is None:
            label = f'AUC{i + 1}({roc_auc:.3f})'
        else:
            label = f'{labels[i]}({roc_auc:.3f})'

        if colors is None:
            ax.plot(recall, precision, label=label, **kwargs)
        else:
            ax.plot(recall, precision, color=colors[i], label=label, **kwargs)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    # ax.axis('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if diag_line:
        ax.plot([0, 1], [1, 0], ls='--', c='grey')

    if title is not None:
        ax.set_title(title)

    if legend_loc is None:
        loc = 'lower left'
    else:
        loc = legend_loc
    ax.legend(loc=loc)

    if grid:
        ax.grid()

    if tight_layout:
        fig.tight_layout()
    return fig, ax


def plot_mul_acc_curve(acc_list, labels=None, thresholds=np.arange(0, 1.01, 0.01), ax=None,
                       legend_kws={}, set_kws={}, grid=True, tight_layout=True, **kwargs):
    if ax is None:
        _, ax = plt.subplots()

    for i, acc in enumerate(acc_list):
        if labels is not None:
            ax.plot(thresholds, acc, label=labels[i], **kwargs)
        else:
            ax.plot(thresholds, acc, **kwargs)

    ax.legend(**legend_kws)
    ax.set_xlim(0, 1)
    ax.set(**set_kws)
    ax.set_xlabel('Thresholds')
    ax.set_ylabel('Accuracy')

    if grid:
        ax.grid()

    if tight_layout:
        plt.tight_layout()

    return ax


def plot_true_network(edges_truth, directed=True, n_nold=None, figsize=(5, 5), **kwargs):
    G = nx.DiGraph() if directed else nx.Graph()

    if n_nold is None:
        n_nold = len(set(edges_truth.flatten()))

    G.add_nodes_from(np.arange(1, n_nold + 1))
    G.add_edges_from(edges_truth)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_circular(G, with_labels=True, font_weight='bold', **kwargs)
    return fig, ax


def plot_predicted_network(edges_truth, edges_pred, directed=True, n_nold=None, figsize=(5, 5),
                           pos_color='g', neg_color='r', misstyle=':', **kwargs):
    set1and2, set1or2, set1not2, set2not1 = two_array_to_edges_set(edges_truth, edges_pred)
    G = nx.DiGraph() if directed else nx.Graph()

    if n_nold is None:
        n_nold = len(set(edges_truth.flatten()))

    G.add_nodes_from(np.arange(1, n_nold + 1))
    G.add_edges_from(set1or2)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_circular(G, with_labels=True, font_weight='bold', edgelist=set1and2,
                     edge_color=pos_color, **kwargs)
    nx.draw_circular(G, with_labels=True, font_weight='bold', edgelist=set1not2,
                     edge_color=neg_color, style=misstyle, **kwargs)
    nx.draw_circular(G, with_labels=True, font_weight='bold', edgelist=set2not1,
                     edge_color=neg_color, **kwargs)
    return fig, ax


def plot_score_matrix(mat, labels=None, annot=True, fmt='.3f', linewidths=1, tight_layout=True,
                      cmap='YlGnBu', tick_bin=1, ticklabel_rotation=0, ax=None, figsize=(6, 5),
                      diag_line=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mat, annot=annot, fmt=fmt, linewidths=linewidths, cmap=cmap, ax=ax, **kwargs)

    if labels is None:
        labels = np.arange(0, len(mat), tick_bin)

    ticks = np.arange(0+.5, len(mat)+.5, tick_bin)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    if len(ticks) == len(labels):
        ax.set_xticklabels(labels, rotation=ticklabel_rotation)
        ax.set_yticklabels(labels, rotation=ticklabel_rotation)

    if diag_line:
        ax.plot([0, len(mat)], [0, len(mat)], c='gray')

    if tight_layout:
        plt.tight_layout()

    return ax


def plot_annot_square(idx, ax=None, **kwargs):
    ys, xs = idx
    if ax is None:
        fig, ax = plt.subplots()

    for x, y in zip(xs, ys):
        ax.plot([x, x, x+1, x+1, x], [y, y+1, y+1, y, y], **kwargs)
    return ax
