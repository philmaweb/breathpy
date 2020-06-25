
from collections import namedtuple, defaultdict, OrderedDict, Counter
from pathlib import Path
import glob
import numpy as np
from numbers import Integral
import pandas as pd
import re
import zipfile
import os
from shutil import copyfile, rmtree
import tempfile
from itertools import zip_longest
from seaborn import color_palette
import warnings
import joblib
import six
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import _criterion
from sklearn.tree import _tree

from sklearn.tree.export import SENTINEL, _color_brew

from ..model.BreathCore import MccImsAnalysis, MccImsMeasurement, PredictionModel
from ..view.BreathVisualizations import RocCurvePlot, ClusterPlot, HeatmapPlot

def get_peax_binary_path():
    """
    Return path where PEAX-binary is located
    """
    # would have expected to be relative to this file, but not the case.
    # path is somehow relative to breathpy dir - odd
    breathpy_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    peax_path = breathpy_path/"bin/peax1.0-LinuxX64/peax"
    return peax_path

def extract_zip(path):
    """Extract a zip-archive, which contains the data files.

    Keyword arguments:
    path -- provide a path of the zip-file, which is supposed to be extracted
    """
    files = zipfile.ZipFile(path, 'r')
    for file in files.namelist():
        files.extract(file, path.rsplit(".", maxsplit=1)[0])



def stratify_selection_by_label(keys, keys_to_label_dict, no_keys_per_class=-1):
    if no_keys_per_class == -1:
        return keys

    if not no_keys_per_class:
        raise ValueError("need at least 1 element per class")
    else:
        labels = sorted(np.unique(list(keys_to_label_dict.values())))
        label_occurences = {l : 0 for l in labels}
        result = []
        for k in keys:
            current_label = keys_to_label_dict[k]
            if label_occurences[current_label] < no_keys_per_class:
                label_occurences[current_label] += 1
                result.append(k)
        return result

def file_limit_stratify_selection_by_label(keys, keys_to_label_dict, labels_to_keep=(), file_limit=-1):
    """
    Stratify class_label dict - get same number of filenames for each label - limit to certain subtypes or limit overall files
    :param keys:
    :param keys_to_label_dict:
    :param labels_to_keep:
    :param file_limit:
    :return:
    """
    if labels_to_keep:
        keys = [fn for fn in keys if keys_to_label_dict[fn] in labels_to_keep]

    if file_limit > 0 and file_limit < len(keys):
        # apply number_of_files_limit so that we have a stratified set - approx equal number of measurements per class

        class_counts = Counter(keys_to_label_dict.values())

        if labels_to_keep:
            for unwanted_label in set(class_counts.keys()).difference(set(labels_to_keep)):
                class_counts.pop(unwanted_label)

        num_of_classes = len(class_counts.keys())
        max_num_of_files_per_class = int(file_limit//num_of_classes)
        keys = stratify_selection_by_label(keys, keys_to_label_dict, max_num_of_files_per_class)
        print(f"Limited each class of {list(class_counts.keys())} to {max_num_of_files_per_class} instances to match requirement of {file_limit} measurements.")
        # return in_file_names
    return keys



def export_graphviz_personalized(decision_tree, out_file=SENTINEL, max_depth=None,
                    feature_names=None, class_names=None, label='all',
                    filled=False, leaves_parallel=False, impurity=True,
                    node_ids=False, proportion=False, rotate=False,
                    rounded=False, special_characters=False, precision=3):
        # tree, decision_tree_model, feature_names, class_labels):
    def export_graphviz(decision_tree, out_file=SENTINEL, max_depth=None,
                        feature_names=None, class_names=None, label='all',
                        filled=False, leaves_parallel=False, impurity=True,
                        node_ids=False, proportion=False, rotate=False,
                        rounded=False, special_characters=False, precision=3):
        """Export a decision tree in DOT format.

        This function generates a GraphViz representation of the decision tree,
        which is then written into `out_file`. Once exported, graphical renderings
        can be generated using, for example::

            $ dot -Tps tree.dot -o tree.ps      (PostScript format)
            $ dot -Tpng tree.dot -o tree.png    (PNG format)

        The sample counts that are shown are weighted with any sample_weights that
        might be present.

        Read more in the :ref:`User Guide <tree>`.

        Parameters
        ----------
        decision_tree : decision tree classifier
            The decision tree to be exported to GraphViz.

        out_file : file object or string, optional (default='tree.dot')
            Handle or name of the output file. If ``None``, the result is
            returned as a string. This will the default from version 0.20.

        max_depth : int, optional (default=None)
            The maximum depth of the representation. If None, the tree is fully
            generated.

        feature_names : list of strings, optional (default=None)
            Names of each of the features.

        class_names : list of strings, bool or None, optional (default=None)
            Names of each of the target classes in ascending numerical order.
            Only relevant for classification and not supported for multi-output.
            If ``True``, shows a symbolic representation of the class name.

        label : {'all', 'root', 'none'}, optional (default='all')
            Whether to show informative labels for impurity, etc.
            Options include 'all' to show at every node, 'root' to show only at
            the top root node, or 'none' to not show at any node.

        filled : bool, optional (default=False)
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        leaves_parallel : bool, optional (default=False)
            When set to ``True``, draw all leaf nodes at the bottom of the tree.

        impurity : bool, optional (default=True)
            When set to ``True``, show the impurity at each node.

        node_ids : bool, optional (default=False)
            When set to ``True``, show the ID number on each node.

        proportion : bool, optional (default=False)
            When set to ``True``, change the display of 'values' and/or 'samples'
            to be proportions and percentages respectively.

        rotate : bool, optional (default=False)
            When set to ``True``, orient tree left to right rather than top-down.

        rounded : bool, optional (default=False)
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        special_characters : bool, optional (default=False)
            When set to ``False``, ignore special characters for PostScript
            compatibility.

        precision : int, optional (default=3)
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.

        Returns
        -------
        dot_data : string
            String representation of the input tree in GraphViz dot format.
            Only returned if ``out_file`` is None.

            .. versionadded:: 0.18

        Examples
        """

        def get_color(value):
            # Find the appropriate color & intensity for a node
            if colors['bounds'] is None:
                # Classification tree
                # get main color from colors - and distort in alpha channel to highlight class purity
                color = list(colors['rgb'][np.argmax(value)])
                sorted_values = sorted(value, reverse=True)
                if len(sorted_values) == 1:
                    alpha = 0
                else:
                    alpha = int(np.round(255 * (sorted_values[0] -
                                                sorted_values[1]) /
                                         (1 - sorted_values[1]), 0))
            else:
                # Regression tree or multi-output
                color = list(colors['rgb'][0])
                alpha = int(np.round(255 * ((value - colors['bounds'][0]) /
                                            (colors['bounds'][1] -
                                             colors['bounds'][0])), 0))

            # Return html color code in #RRGGBBAA format
            color.append(alpha)
            hex_codes = [str(i) for i in range(10)]
            hex_codes.extend(['a', 'b', 'c', 'd', 'e', 'f'])
            color = [hex_codes[c // 16] + hex_codes[c % 16] for c in color]

            return '#' + ''.join(color)


        def node_to_str(tree, node_id, criterion):
            # Generate the node content string
            if tree.n_outputs == 1:
                value = tree.value[node_id][0, :]
            else:
                value = tree.value[node_id]

            # Should labels be shown?
            labels = (label == 'root' and node_id == 0) or label == 'all'

            # PostScript compatibility for special characters
            if special_characters:
                characters = ['&#35;', '<SUB>', '</SUB>', '&le;', '<br/>', '>']
                node_string = '<'
            else:
                characters = ['#', '[', ']', '<=', '\\n', '"']
                node_string = '"'

            # Write node ID
            if node_ids:
                if labels:
                    node_string += 'node '
                node_string += characters[0] + str(node_id) + characters[4]

            # Write decision criteria
            if tree.children_left[node_id] != _tree.TREE_LEAF:
                # Always write node decision criteria, except for leaves
                if feature_names is not None:
                    feature = feature_names[tree.feature[node_id]]
                else:
                    feature = "X%s%s%s" % (characters[1],
                                           tree.feature[node_id],
                                           characters[2])
                node_string += '%s %s %s%s' % (feature,
                                               characters[3],
                                               round(tree.threshold[node_id],
                                                     precision),
                                               characters[4])

            # Write impurity
            if impurity:
                if isinstance(criterion, _criterion.FriedmanMSE):
                    criterion = "friedman_mse"
                elif not isinstance(criterion, six.string_types):
                    criterion = "impurity"
                if labels:
                    node_string += '%s = ' % criterion
                node_string += (str(round(tree.impurity[node_id], precision)) +
                                characters[4])

            # Write node sample count
            if labels:
                node_string += 'samples = '
            if proportion:
                percent = (100. * tree.n_node_samples[node_id] /
                           float(tree.n_node_samples[0]))
                node_string += (str(round(percent, 1)) + '%' +
                                characters[4])
            else:
                node_string += (str(tree.n_node_samples[node_id]) +
                                characters[4])

            # Write node class distribution / regression value
            if proportion and tree.n_classes[0] != 1:
                # For classification this will show the proportion of samples
                value = value / tree.weighted_n_node_samples[node_id]
            if labels:
                node_string += 'value = '
            if tree.n_classes[0] == 1:
                # Regression
                value_text = np.around(value, precision)
            elif proportion:
                # Classification
                value_text = np.around(value, precision)
            elif np.all(np.equal(np.mod(value, 1), 0)):
                # Classification without floating-point weights
                value_text = value.astype(int)
            else:
                # Classification with floating-point weights
                value_text = np.around(value, precision)
            # Strip whitespace
            value_text = str(value_text.astype('S32')).replace("b'", "'")
            value_text = value_text.replace("' '", ", ").replace("'", "")
            if tree.n_classes[0] == 1 and tree.n_outputs == 1:
                value_text = value_text.replace("[", "").replace("]", "")
            value_text = value_text.replace("\n ", characters[4])
            node_string += value_text + characters[4]

            # Write node majority class at the top of the box
            if (class_names is not None and
                    tree.n_classes[0] != 1 and
                    tree.n_outputs == 1):
                # Only done for single-output classification trees
                if labels:
                    node_string += 'class = '
                if class_names is not True:
                    class_name = class_names[np.argmax(value)]
                else:
                    class_name = "y%s%s%s" % (characters[1],
                                              np.argmax(value),
                                              characters[2])
                node_string += class_name

            # Clean up any trailing newlines
            if node_string[-2:] == '\\n':
                node_string = node_string[:-2]
            if node_string[-5:] == '<br/>':
                node_string = node_string[:-5]

            return node_string + characters[5]


        def personalized_node_to_str(tree, node_id, criterion):
            try:
                feature = feature_names[tree.feature[node_id]]
            except IndexError as ie:
                feature = feature_names[0]
                print(f"Index error while building decision tree visualization with index {tree.feature[node_id]} on {feature_names}")

            # Should labels be shown?
            labels = (label == 'root' and node_id == 0) or label == 'all'

            feature_str = ""
            if tree.children_left[node_id] != _tree.TREE_LEAF:
                # eg Peak_0460 <= 79.259
                feature_str = '{} {} {}'.format(feature, '&le;', round(tree.threshold[node_id], precision))

            characters = ['#', '[', ']', '<=', '\\n', '"']

            samples_string = ""
            # Write node sample count
            if labels:
                samples_string += 'samples = '

            if proportion:
                percent = (100. * tree.n_node_samples[node_id] /
                           float(tree.n_node_samples[0]))
                samples_string += (str(round(percent, 1)) + '%')  # + characters[4])
            else:
                samples_string += (str(tree.n_node_samples[node_id]))  # + characters[4])

            # array of size len(n_classes)
            node_value = tree.value[node_id][0, :]

            # Write node majority class
            class_str = ""
            # import pdb;pdb.set_trace()
            if (class_names is not None and
                    tree.n_classes[0] != 1 and
                    tree.n_outputs == 1
            ):
                # Only done for single-output classification trees
                if labels:
                    class_name = class_names[np.argmax(node_value)]
                    class_str = 'class = {}'.format(class_name)

            return feature_str, node_value, class_str, samples_string


        def personalized_value_formater(node_value):
            # round to 100 percent steps for bar represetation in table
            sample_count = sum(node_value)
            # factor = sample_count / 10
            factor = sample_count / 100
            formated = np.round(node_value / factor)
            return np.asarray(formated, dtype=int)

        def clamp(x):
            return max(0, min(x, 255))

        def rgb_to_hex(rgb_lis):
            return "#{0:02x}{1:02x}{2:02x}".format(clamp(rgb_lis[0]), clamp(rgb_lis[1]), clamp(rgb_lis[2]))

        def generate_color_bar_table(formated_node_value, pure_class_colors):
            # n_classes = len(formated_node_value)
            row_str = "<TR>"
            td_lis = []
            for colspan, class_color in zip(formated_node_value, pure_class_colors):
                # colspan cannot be 0 - skip column if no occurence
                if colspan:
                    td_lis.append(f"<TD colspan='{colspan}' bgcolor='{rgb_to_hex(class_color)}' border='1'></TD>")
            row_str += "\n".join(td_lis)
            row_str += "</TR>\n"
            hundred_tds_row = f"<TR>{'<TD></TD>'*100}</TR>\n"
            # ten_tds_row = """<TR><TD></TD><TD></TD><TD></TD><TD></TD><TD></TD><TD></TD><TD></TD><TD></TD><TD></TD><TD></TD></TR>\n"""
            # row_str += ten_tds_row
            row_str += hundred_tds_row
            return row_str

        def get_pure_colors(n):
            """
            Get custom color - use seaborn standard palette if n < 11, otherwise _color_brew is used
            :param n:
            :return:
            """
            if n == 1 or n > 10:
                pure_classes_colors = _color_brew(n)
            else:
                seaborn_colors = color_palette()[:n]
                # convert to rgb ints - seaborn uses floats instead of ints
                pure_classes_colors = [[int(i * 255) for i in e] for e in seaborn_colors]
            return  pure_classes_colors

        def recurse(tree, node_id, criterion, parent=None, depth=0):
            if node_id == _tree.TREE_LEAF:
                raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]

            n_classes = len(class_names)
            # colors for color bar at bottom - matches fill-color
            pure_classes_colors = get_pure_colors(n_classes)

            # Add node with description
            if max_depth is None or depth <= max_depth:

                # Collect ranks for 'leaf' option in plot_options
                if left_child == _tree.TREE_LEAF:
                    ranks['leaves'].append(str(node_id))
                elif str(depth) not in ranks:
                    ranks[str(depth)] = [str(node_id)]
                else:
                    ranks[str(depth)].append(str(node_id))

                feature_str, node_value, class_str, samples_str = personalized_node_to_str(tree, node_id, criterion)
                formated_node_value = personalized_value_formater(node_value)
                colored_rows = generate_color_bar_table(formated_node_value=formated_node_value, pure_class_colors=pure_classes_colors)

                samples_explanation_strs = []
                for i in range(len(class_names)):
                    class_name = class_names[i]
                    class_count = int(node_value[i])

                    samples_explanation_strs.append(f"{class_name} : {class_count}")
                    # add newline every 2 classes

                # formated_samples_explanation_str = ", ".join(samples_explanation_strs)
                #
                # bar_str = '''<
                # <TABLE border='0' cellborder='0' cellspacing='0' cellpadding='7'>
                #     <TR><TD colspan='10'>{}</TD></TR>
                #     <TR><TD colspan='10'>{}</TD></TR>
                #     <TR><TD colspan='10'>{}</TD></TR>
                #     <TR><TD colspan='10'>{}</TD></TR>
                #     {}
                # </TABLE>>'''.format(feature_str, class_str, samples_str, formated_samples_explanation_str, colored_rows)
                #
                # adjust for 100 cols for more accurate representation
                formated_samples_explanation_str = ""
                current_sample_expl = ""

                # add new row so we only have 2 class_label per row
                row_building_block_start = "<TR><TD colspan='100'>"
                row_building_block_end = "</TD></TR>"
                it = iter(samples_explanation_strs)
                for desc_1, desc_2 in zip_longest(it,it, fillvalue=''):
                    sep = ", "
                    if not desc_2:
                        sep = ''
                    formated_samples_explanation_str += f"{row_building_block_start}{desc_1}{sep}{desc_2}{row_building_block_end}"


                bar_str = f"<" \
                        f"<TABLE border='0' cellborder='0' cellspacing='0' cellpadding='1'> <TR>" \
                            f"<TD colspan='100'>{feature_str}</TD></TR>" \
                            f"<TR><TD colspan='100'>{class_str}</TD></TR>" \
                            f"<TR><TD colspan='100'>{samples_str}</TD></TR>" \
                            f"{formated_samples_explanation_str}" \
                            f"{colored_rows}" \
                        f"</TABLE>" \
                    f">"


                # out_file.write('%d [label=%s'
                #                % (node_id,
                #                   node_to_str(tree, node_id, criterion)))
                out_file.write('%d [label=%s'
                               % (node_id,
                                  bar_str))

                if filled:
                    # Fetch appropriate color for node
                    if 'rgb' not in colors:
                        # Initialize colors and bounds if required
                        # Colors for the background
                        colors['rgb'] = get_pure_colors(tree.n_classes[0])
                        if tree.n_outputs != 1:
                            # Find max and min impurities for multi-output
                            colors['bounds'] = (np.min(-tree.impurity),
                                                np.max(-tree.impurity))
                        elif (tree.n_classes[0] == 1 and
                              len(np.unique(tree.value)) != 1):
                            # Find max and min values in leaf nodes for regression
                            colors['bounds'] = (np.min(tree.value),
                                                np.max(tree.value))
                    if tree.n_outputs == 1:
                        # our case
                        node_val = (tree.value[node_id][0, :] /
                                    tree.weighted_n_node_samples[node_id])
                        if tree.n_classes[0] == 1:
                            # Regression
                            node_val = tree.value[node_id][0, :]
                    else:
                        # If multi-output color node by impurity
                        node_val = -tree.impurity[node_id]
                    out_file.write(', fillcolor="%s"' % get_color(node_val))
                out_file.write('] ;\n')

                # if parent is not None:
                #     # Add edge to parent
                #     out_file.write('%d -> %d' % (parent, node_id))
                #     import pdb; pdb.set_trace()
                #     if parent == 0:
                #         # Draw True/False labels if parent is root node
                #         angles = np.array([45, -45]) * ((rotate - .5) * -2)
                #         out_file.write(' [labeldistance=2.5, labelangle=')
                #         if node_id == 1:
                #             out_file.write('%d, headlabel="True"]' % angles[0])
                #         else:
                #             out_file.write('%d, headlabel="False"]' % angles[1])
                #     out_file.write(' ;\n')

                if left_child != _tree.TREE_LEAF:
                    recurse(tree, left_child, criterion=criterion, parent=node_id,
                            depth=depth + 1)

                    out_file.write('%d -> %d' % (node_id, left_child))
                    angles = np.array([45, -45]) * ((rotate - .5) * -2)
                    out_file.write('[labeldistance=2.5, labelangle={}, headlabel="Yes"] ;\n'.format(angles[0]))
                    recurse(tree, right_child, criterion=criterion, parent=node_id,
                            depth=depth + 1)

                    out_file.write('%d -> %d' % (node_id, right_child))
                    out_file.write('[labeldistance=2.5, labelangle={}, headlabel="No"] ;\n'.format(angles[1]))

            else:
                ranks['leaves'].append(str(node_id))

                out_file.write('%d [label="(...)"' % node_id)
                if filled:
                    # color cropped nodes grey
                    out_file.write(', fillcolor="#C0C0C0"')
                out_file.write('] ;\n' % node_id)

                if parent is not None:
                    # Add edge to parent
                    out_file.write('%d -> %d ;\n' % (parent, node_id))

        check_is_fitted(decision_tree, 'tree_')
        own_file = False
        return_string = False
        try:
            if out_file == SENTINEL:
                warnings.warn("out_file can be set to None starting from 0.18. "
                              "This will be the default in 0.20.",
                              DeprecationWarning)
                out_file = "tree.dot"

            if isinstance(out_file, six.string_types):
                if six.PY3:
                    out_file = open(out_file, "w", encoding="utf-8")
                else:
                    out_file = open(out_file, "wb")
                own_file = True

            if out_file is None:
                return_string = True
                out_file = six.StringIO()

            if isinstance(precision, Integral):
                if precision < 0:
                    raise ValueError("'precision' should be greater or equal to 0."
                                     " Got {} instead.".format(precision))
            else:
                raise ValueError("'precision' should be an integer. Got {}"
                                 " instead.".format(type(precision)))

            # Check length of feature_names before getting into the tree node
            # Raise error if length of feature_names does not match
            # n_features_ in the decision_tree
            if feature_names is not None:
                if len(feature_names) != decision_tree.n_features_:
                    raise ValueError("Length of feature_names, %d "
                                     "does not match number of features, %d"
                                     % (len(feature_names),
                                        decision_tree.n_features_))

            # The depth of each node for plotting with 'leaf' option
            ranks = {'leaves': []}
            # The colors to render each node with
            colors = {'bounds': None}

            out_file.write('digraph Tree {\n')

            # Specify node aesthetics
            out_file.write('node [shape=box')
            rounded_filled = []
            if filled:
                rounded_filled.append('filled')
            if rounded:
                rounded_filled.append('rounded')
            if len(rounded_filled) > 0:
                out_file.write(', style="%s", color="black"'
                               % ", ".join(rounded_filled))
            if rounded:
                out_file.write(', fontname=helvetica')
            out_file.write('] ;\n')

            # Specify graph & edge aesthetics
            if leaves_parallel:
                out_file.write('graph [ranksep=equally, splines=polyline] ;\n')
            if rounded:
                out_file.write('edge [fontname=helvetica] ;\n')
            if rotate:
                out_file.write('rankdir=LR ;\n')

            # Now recurse the tree and add node & edge attributes
            if isinstance(decision_tree, _tree.Tree):
                recurse(decision_tree, 0, criterion="impurity")
            else:
                recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

            # If required, draw leaf nodes at same depth as each other
            if leaves_parallel:
                for rank in sorted(ranks):
                    out_file.write("{rank=same ; " +
                                   "; ".join(r for r in ranks[rank]) + "} ;\n")
            out_file.write("}")

            if return_string:
                return out_file.getvalue()

        finally:
            if own_file:
                out_file.close()

    return export_graphviz(
            decision_tree,
            feature_names=feature_names,
            out_file=None, class_names=class_names, proportion=False, filled=True, leaves_parallel=False,
            impurity=False, rounded=False, )


# ##########################
# Functions related to computation of overlap between VOCs
def parse_all_excel(dir, sheetname=None, skiprows=0):
    """
    Read in all excel files in directory (no trailing /) and return dictionary
    {sheet_name : pd_Dataframe} for each sheet
    If sheetname is specified return only dataframes of that sheet
    :param dir:
    :param sheetname:
    :param skiprows:
    :return:
    """
    all_files = glob.glob("{}/*.xls".format(dir))
    return_frames = []
    for f in all_files:
        # use sheetname=None - results in sheets being ordered by sheet names in dict
        excel_dict = pd.read_excel(f, sheetname=None, skiprows=skiprows)
        # sheetname syntax does not work relyably when directly passed - so we parse all sheets and select from that result
        if sheetname and sheetname in excel_dict:
            excel_dict = excel_dict[sheetname]
        return_frames.append(excel_dict)
    return return_frames

def get_all_excel_masterlayers(dir):
    return parse_all_excel(dir=dir, sheetname="layer", skiprows=3)


# find peak mapping from master -> Annotation
# we only want Peaks that are not P<int>
def test_peak_unkown():
    text = "P0"
    text2 = "P10"
    text3 = "P1b"
    text4 = "Methanol"
    text_lis = [text, text2, text3, text4]
    for t in text_lis:
        regex = re.findall(r'^P\d*',t)
        if(regex):
            print("found {} in {}".format(regex, t))

def is_peak_unknown(string):
    return bool(re.findall(r'^P\d*',string.strip()))

def remove_unkown_peaks(voc_frame):
    known_peaks_mask = [not is_peak_unknown(v) for v in voc_frame['Name'].values]
    return voc_frame.loc[known_peaks_mask]

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def to_rect(x, y, rx, ry):
    left = x - rx
    right = x + rx
    top = y + ry
    bottom = y - ry
    return (Rectangle(left, bottom, right, top))


def area(a, b):  # returns 0.0 if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    return 0.0


def calculate_overlap(x, y, w, h, x2, y2, w2, h2):
    """
    calculate overlap between two peak windows as percentage of the smaller peak are
    """
    area1 = w * h * 4
    area2 = w2 * h2 * 4
    smallest_area = min(area1, area2)

    a = to_rect(x, y, w, h)
    b = to_rect(x2, y2, w2, h2)

    overlap = area(a, b)
    if overlap > 0.0 and smallest_area > 0.0:
        percentage = round(100 * overlap / smallest_area, 2)
        #         print("overlap is {} = {}%".format(overlap, percentage))
        return percentage
    else:
        return overlap

def test_calculate_orverlap():
    p1 = (0.549, 8.1, 0.003, 2.6)
    p2 = (0.549, 6.9, 0.003, 3.3)

    assert(calculate_overlap(1,1,1,1, 1,1,1,1) == 1.0)
    assert(calculate_overlap(1,1,1,1, 3,3,1,1) == 0.0)
    print( calculate_overlap(*p1, *p2))

def overlap_mapper(layer1, layer2):
    my_dict = defaultdict(list)
    for i, master_row in layer1.iterrows():
        m_name, m_c, m_k0, m_rt, m_k0r, m_rtr = master_row.values[:6]
        for j, voc_row in layer2.iterrows():
            v_name, v_c, v_k0, v_rt, v_k0r, v_rtr = voc_row.values[:6]

            o = calculate_overlap(m_rt, m_k0, m_rtr, m_k0r, v_rt, v_k0, v_rtr, v_k0r)
            if o > 0.0:
                my_dict[m_name].append((v_name, o))
    return my_dict

###################
# Functions related to class label management

def unite_class_label_dict_in_dir(dirname):
    full_class_label_dict = dict()
    dict_fns = glob.glob(f"{dirname}/*_class_labels.csv")


    for dict_fn in dict_fns:
        full_class_label_dict.update(MccImsAnalysis.parse_class_labels(dict_fn))

    return full_class_label_dict


def filter_class_label_dict_to_present_names(dirname, class_label_dict, file_suffix="*_ims.csv"):
    present_measurements = glob.glob(f"{dirname}/{file_suffix}")

    present_measurements = [Path(pm).stem + Path(pm).suffix for pm in present_measurements]
    filtered_dict = {}

    for m in present_measurements:
        if m in class_label_dict:
            filtered_dict[m] = class_label_dict[m]
    #sort by filename
    if not len(list(filtered_dict.keys())) == len(present_measurements):
        missing_names = set(list(filtered_dict.keys())).difference(present_measurements)
        print(f"class_labels dont match measurements - missing class labels for {missing_names}")
        # import ipdb; ipdb.set_trace()
    return OrderedDict(sorted(filtered_dict.items(), key=lambda t: t[0]))


#################
# Functions related to dataset + paper preparation

def prepare_peak_detection_results_zip(file_params, preprocessing_steps):
    """
    Create a zip archive for peak detection results with up to date class labels and subfolders of the pdr by pdr method
    :param dirname:
    :return:
    """
    data_dir = file_params['folder_path']
    results_dir = file_params['out_dir']
    # filter out peak_detection_methods
    from .model.BreathCore import PeakDetectionMethod, ExternalPeakDetectionMethod
    pdms = []
    for prep_step in preprocessing_steps:
        try:
            pdm = PeakDetectionMethod(prep_step)
            pdms.append(pdm)
        except ValueError as Ve:
            pass
        try:
            pdm = ExternalPeakDetectionMethod(prep_step)
            pdms.append(pdm)
        except ValueError as ve:
            pass

    # make sure all filename match in raw_dir / data dir
    if "candy" in file_params['folder_name']:
        # copy master dict to raw dir
        master_candy_file = "/home/philipp/dev/breathpy/data/full_candy/class_labels.csv"
        copyfile(master_candy_file, Path(data_dir).joinpath("full_candy_class_labels.csv"))

    filtered_class_label_dict_fn = prepare_adjust_class_label_dict_to_measurements(data_dir)


    # for each pdm make subfolder
    for pdm in pdms:
        pdm_name = pdm.name
        subfolder_name = file_params['folder_name']
        subfolder_pdm = "_".join([subfolder_name, pdm_name, "results"])

        new_result_dir = Path(data_dir)

        new_result_dir = new_result_dir.joinpath(Path(subfolder_pdm))

        # list all pdr in results dir
        glob_str = f"{results_dir}*{pdm_name}*_peak_detection_result.csv"
        all_results_by_pdm = glob.glob(glob_str)

        new_result_dir.mkdir(exist_ok=True)

        for rfn in all_results_by_pdm:
            rfn_stem = Path(rfn).stem + Path(rfn).suffix
            copyfile(rfn, Path(new_result_dir).joinpath(rfn_stem))

        class_label_fn = MccImsAnalysis.guess_class_label_extension(data_dir)
        class_label_fn_stemmed = Path(class_label_fn).stem + Path(class_label_fn).suffix
        copyfile(Path(class_label_fn), Path(new_result_dir).joinpath(class_label_fn_stemmed))

    # class_label dict should be in data_dir
    # create correct subfolder by pdm
    # should copy results by peak_dM ending to correct subfolder
    # now manually create zips and be done


def prepare_adjust_class_label_dict_to_measurements(dirname):
    full_class_label_dict = unite_class_label_dict_in_dir(dirname)
    f_dict = filter_class_label_dict_to_present_names(dirname, full_class_label_dict)

    out_name = "filtered_class_labels.csv"
    out_path = f"{dirname}/{out_name}"
    if full_class_label_dict:
        with open(out_path, "w") as fh:
            fh.write("name,label\n")
            for k, v in f_dict.items():
                fh.write(f"{k},{v}\n")
    print("Finished writing filtered class_labels")
    return out_path


#################
# Paper related plots

def plot_probe_clustering_grid(mcc_ims_measurement, plot_parameters):
    from view.BreathVisualizations import HeatmapPlot, ClusterPlot, save_plot_to_buffer
    import matplotlib.pyplot as plt

    title = "Grid Plot"
    plot_dir_suffix="heatmaps"
    plot_type="grid_plot"
    intensity_matrix = mcc_ims_measurement.df

    figure_dir = '{}{}/'.format(plot_parameters['plot_dir'], plot_dir_suffix)

    # ClusterPlot._plot_overlay_clustering_helper(
    #     intensity_matrix=mcc_ims_measurement.df, pdm_coord_tuples=peak_alignment_result.peak_coordinates.items(),
    #     plot_parameters=plot_parameters, title_dict=title_dict,
    #     best_features_df=mcc_ims_analysis.analysis_result.best_features_df)
    ax, fig, irm_vals, rt_vals = ClusterPlot._setup_overlay_fast(intensity_matrix, plot_parameters, figure_dir)

    peak_coords_map = MccImsAnalysis._compute_peak_id_coord_map(threshold_inverse_reduced_mobility=0.015, threshold_scaling_retention_time=0.1)
    peak_coord_df = pd.DataFrame(peak_coords_map).T

    scaled_coords = ClusterPlot._add_scaled_coords_fast(irm_vals, rt_vals, peak_coord_df)
    # pdb.set_trace()
    rects = ClusterPlot._construct_coord_rects(scaled_coords, color="black")
    # cluster_centers = [(row['inverse_reduced_mobility'], row['retention_time']) for _, row in new_scaled_coords.iterrows()]
    added_rects = [ax.add_patch(rect) for rect in rects]

    # Scatterplot to mark peak positions
    # ax.scatter(data=new_scaled_coords, x="scaled_irm", y="scaled_rt", marker='*')
    # title_string = "Peaks in {}, {}".format(title_dict['measurement_name'],
    #                                         title_dict['peak_detection_method_name'])

    # ax.set_title(title_string)
    # figure_name = "{}_overlay_{}_{}.png".format(title_dict['dataset_name'],
    #                                             title_dict['peak_detection_method_name'],
    #                                             title_dict['measurement_name'])

    # if not plot_parameters.get('use_buffer', False):
    #     fig.savefig("{}{}".format(figure_dir, figure_name), dpi=300, bbox_inches='tight', format="png")
    # return_figures.append(
    #     (pdmn, title_dict['measurement_name'], save_plot_to_buffer(plot_parameters, fig), figure_name))
    # remove rects for next iteration
    # for rect in added_rects: rect.remove()


    # title_string = title
    #
    # plt.yticks(rotation=0)
    # ax.set_title(title_string)


    plot_prefix = ''
    if 'plot_prefix' in plot_parameters:
        plot_prefix = plot_parameters['plot_prefix']
        plot_prefix = "fast_{}".format(plot_prefix)
        # import pdb;pdb.set_trace()
    dataset_name = plot_prefix

    figure_name = '{}_{}.png'.format(dataset_name, plot_type)


    if not plot_parameters.get('use_buffer', False):
        Path(figure_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving figure to {figure_dir}{figure_name}")
        fig.savefig("{}{}".format(figure_dir, figure_name), dpi=200, bbox_inches='tight', format="png",
                    compress_level=1)

    return_figure = save_plot_to_buffer(plot_parameters, fig)
    plt.close()
    return return_figure

def make_example_paper_plots(file_params, plot_params, preprocessing_steps, evaluation_params):
    """
    Create plots for paper - Raw - preprocessed - cluster - overlay plot - ROC plot - Boxplot
    :param file_params:
    :param plot_params:
    :param preprocessing_steps:
    :param evaluation_params:
    :return:
    """
    # pseudog dataset looks very unimpressive raw / processed

    full_path_to_in_files = glob.glob(file_params['folder_path'] + "*_ims.csv")

    # label_dict = MccImsAnalysis.parse_class_labels(file_params['label_filename'])
    label_dict_path = MccImsAnalysis.guess_class_label_extension(file_params['folder_path'])
    label_dict = MccImsAnalysis.parse_class_labels(label_dict_path)

    visualnow_layer_path = [filename for filename in glob.glob(file_params['folder_path'] + "*") if
                        (str.endswith(filename, "layer.csv") or str.endswith(filename, "layer.xls"))][0]

    in_file_names = [fp.rsplit("/", maxsplit=1)[-1] for fp in full_path_to_in_files]

    # check if output directory already exists
    if not Path(file_params['out_dir']).exists():
        # create directory if it doesnt exist already
        Path(file_params['out_dir']).mkdir()

    # check if already peaxed, if yes, then read in peaxed files
    outfile_names = [file_params['out_dir'] + fn[:-4] + "_out.csv" for fn in in_file_names]
    # are_all_files_already_peaxed = all([Path(on).exists() for on in outfile_names])

    print("Extracting CSV files {}".format(in_file_names))
    # Use parse raw measurement instead of extracting from zip everytime
    # my_csv_df = extract_csvs_from_zip(raw_zip_archive, verbose=True)
    sample_ims_measurements = [MccImsMeasurement(f"{file_params['folder_path']}{i_fn}") for i_fn in in_file_names[:5]]

    # weird index problem fixed: (123,) instead of single number
    # my_ims_measurements = [MccImsMeasurement(i_fn) for i_fn in full_path_to_in_files]
    print("Finished Extracting CSV files.\n")
    print(sample_ims_measurements)

    # for m in my_ims_measurements:
    #     print(m.filename, m.header['comment'])
    # pdb.set_trace()

    peax_binary_path = "{0}bin/peax1.0-LinuxX64/peax".format(file_params['dir_level'])

    ims_analysis = MccImsAnalysis(sample_ims_measurements, preprocessing_steps, outfile_names,
                                  performance_measure_parameters=evaluation_params,
                                  class_label_file=label_dict_path,
                                  dir_level=file_params['dir_level'],
                                  dataset_name=file_params['folder_name'],
                                  visualnow_layer_file=visualnow_layer_path,
                                  peax_binary_path=peax_binary_path
                                   )

    ims_analysis.measurements = sample_ims_measurements[:5].copy()
    # plot_raw_and_processed(ims_analysis, plot_parameters)

    # make probe clustering plot to show grid
    # sample measurements are corrupted - or at least not normalized
    from tools.tools import plot_probe_clustering_grid
    grid_measurement = MccImsMeasurement(raw_filename="/home/philipp/dev/breathpy/data/small_candy_anon/BD18_1408280838_ims.csv")
    grid_measurement.normalize_by_intensity()
    # grid_measurement.denoise_crop_inverse_reduced_mobility()
    plot_probe_clustering_grid(grid_measurement, plot_parameters=plot_params)
    # HeatmapPlot.FastIntensityMatrix(grid_measurement, plot_parameters=plot_parameters)

    # ims_analysis.measurements = my_ims_measurements
    ims_analysis = MccImsAnalysis([], preprocessing_steps, outfile_names,
                                  performance_measure_parameters=evaluation_params,
                                  dir_level=file_params['dir_level'],
                                  dataset_name=file_params['folder_name'],
                                  class_label_file=label_dict_path,
                                  visualnow_layer_file=visualnow_layer_path,
                                  peax_binary_path=peax_binary_path)

    ims_analysis.import_results_from_csv_dir(file_params['out_dir'], class_label_file=label_dict_path)

    # save peak detection results with peak detection name and _out.csv
    ims_analysis.align_peaks(file_params['file_prefix'])

    #  make probe clustering demo plot
    ClusterPlot._plot_overlay_clustering_helper_fast(intensity_matrix=ims_analysis.measurements[0].df, pdm_coord_tuples=ims_analysis.peak_alignment_result.peak_coordinates.items(), plot_parameters=plot_params, title_dict={}, best_features_df=ims_analysis.analysis_result.best_features_df)

    # ClusterPlot.ClusterBasic(ims_analysis, plot_parameters=plot_params)
    ClusterPlot.ClusterMultiple(ims_analysis, plot_parameters=plot_params)
    # ClusterPlot.OverlayAlignment(ims_analysis, plot_parameters=plot_params)

    # ClusterPlot.OverlayClasswiseAlignment(ims_analysis, plot_parameters=plot_params)
    # import pdb; pdb.set_trace()

    ims_analysis.reduce_features(ims_analysis.AVAILABLE_FEATURE_REDUCTION_METHODS)
    ims_analysis.evaluate_performance()
    # get best model
    best_model_name, feature_names, decision_tree_buffer = ims_analysis.get_best_model()

    # ClusterPlot.OverlayBestFeaturesAlignment(ims_analysis, plot_parameters=plot_params)
    # ims_analysis.analysis_result.export_statistics()
    # make sure cross val was run - minimum

    if ims_analysis.is_able_to_cross_validate():
        if len(set(ims_analysis.analysis_result.class_labels)) == 2:
            RocCurvePlot.ROCCurve(ims_analysis.analysis_result, plot_parameters=plot_params)
        else:
            RocCurvePlot.MultiClassROCCurve(ims_analysis.analysis_result, plot_parameters=plot_params)
    # BoxPlot.BoxPlotBestFeature(ims_analysis.analysis_result, plot_parameters=plot_params)
    # TreePlot.DecisionTrees(ims_analysis.analysis_result, plot_parameters=plot_params)
    #
    # ClusterPlot.OverlayBestFeaturesClasswiseAlignment(ims_analysis, plot_parameters=plot_parameters)
    # ClusterPlot.OverlayBestFeaturesAlignment(ims_analysis, plot_parameters=plot_parameters)


    tmp = os.path.join(tempfile.gettempdir(), '.breath/{}'.format(hash(os.times())))
    os.makedirs(tmp)


    # test_set_path = "../data/test_full_candy/"
    # test_labels_path = f"{test_set_path}class_labels.csv"
    # test_names = "BD18_1711291652_ims.csv BD18_1711291656_ims.csv BD18_1711291659_ims.csv BD18_1711291702_ims.csv BD18_1711291705_ims.csv BD18_1711291709_ims.csv BD18_1711291712_ims.csv BD18_1711291715_ims.csv BD18_1711291719_ims.csv BD18_1711291722_ims.csv".split()
    # test_names = ['BD18_1511121702_ims.csv', 'BD18_1711291712_ims.csv', 'BD18_1711291756_ims.csv', 'BD18_1408280844_ims.csv', 'BD18_1711291646_ims.csv', 'BD18_1711291722_ims.csv', 'BD18_1711291709_ims.csv', 'BD18_1711291725_ims.csv', 'BD18_1511121719_ims.csv']
    # test_paths = ["{}{}".format(test_set_path, test_name) for test_name in test_names]
    dataset_name = file_params['folder_path'].split("/")[-2]
    if dataset_name.startswith("train_"):
        test_dir = file_params['folder_path'].replace("train_", "test_")
    else:
        test_dir = file_params['folder_path']

    test_full_path_to_in_files = glob.glob(test_dir + "*_ims.csv")
    test_in_file_names = [fp.split("/")[-1] for fp in test_full_path_to_in_files]

    test_labels_dict = MccImsAnalysis.parse_class_labels(MccImsAnalysis.guess_class_label_extension(test_dir))
    filtered_stratified_test_names = file_limit_stratify_selection_by_label(test_in_file_names,
                                                                            keys_to_label_dict=test_labels_dict,
                                                                            )

    test_paths = [f"{test_dir}{i_fn}" for i_fn in filtered_stratified_test_names]
    test_measurements = [MccImsMeasurement(tp) for tp in test_paths]

    predictor_path = tmp + "/pred_model.sav"
    ims_analysis.analysis_result.export_prediction_models(path_to_save=predictor_path)
    predictors = joblib.load(predictor_path)

    # clean up tempdir
    rmtree(tmp, ignore_errors=True)

    predictionModel = PredictionModel(
        preprocessing_params={s: {} for s in preprocessing_steps},
        evaluation_params=ims_analysis.performance_measure_parameter_dict,
        scipy_predictor_by_pdm=predictors,
        feature_names_by_pdm=ims_analysis.analysis_result.feature_names_by_pdm,
        peax_binary_path=peax_binary_path,
        visualnow_layer_file=visualnow_layer_path)

    prediction = predictionModel.predict(test_measurements)
    # is always sorted

    class_labels = np.unique([m.class_label for m in ims_analysis.measurements])
    for pdm, prediction_index in prediction.items():
        predicted_labels = {test_name: class_labels[p] for p, test_name in
                            zip(prediction_index, filtered_stratified_test_names)}
        correct = dict()
        false = dict()
        for fn, predicted_label in predicted_labels.items():
            if predicted_label == test_labels_dict[fn]:
                correct[fn] = predicted_label
            else:
                false[fn] = predicted_label

        print("resulting_labels for {} are: {}".format(pdm.name, predicted_labels))
        print("Falsely classified: {}".format(false))
        print("That's {} correct vs {} false".format(len(correct.keys()), len(false.keys())))

def load_plot_feature_matrix(fm_filename, class_label_fn, plot_parameters):
    """
    Plot a feature matrix from file and split by class labels
    :param fm_filename:
    :param class_label_fn:
    :param plot_parameters:
    :return:
    """
    fm = pd.read_csv(fm_filename, sep=",", index_col="Measurement")
    class_label_dict = MccImsAnalysis.parse_class_labels(class_label_fn)
    outname = f"{fm_filename}_fm.png"
    HeatmapPlot.FeatureMatrixPlot(fm, class_label_dict, plot_parameters, outname)


def compare_peax_results(fn1, fn2, df1=None, df2=None):
    # read in
    # sort by retention time, then irm
    # compare irm val + intensity
    # peak id - don't care
    if df1 is None:
        try:
            df1 = pd.read_csv(fn1, sep="\t", index_col=0)
            df1 = df1.sort_values(by=['retention_time', 'inverse_reduced_mobility'])[['retention_time', 'inverse_reduced_mobility', 'intensity']].reindex()
        except KeyError:  # we have peax raw result
            df1 = pd.read_csv(fn1, sep="\t", index_col=0)
            df1 = df1.rename(columns={"r":'retention_time', "t":'inverse_reduced_mobility', "signal":'intensity'})
    df1 = df1.sort_values(by=['retention_time', 'inverse_reduced_mobility'])[
                ['retention_time', 'inverse_reduced_mobility', 'intensity']].reindex()
    if df2 is None:
        df2 = pd.read_csv(fn2, sep="\t", index_col=0)
    df2 = df2.sort_values(by=['retention_time', 'inverse_reduced_mobility'])[['retention_time', 'inverse_reduced_mobility', 'intensity']].reindex()

    # now compare
    comp_lis= []
    for i in range(df1.shape[0]):
        row1 = df1.iloc[[i]]
        row2 = df2.iloc[[i]]
        row_comp = all(row1 == row2)
        if not row_comp:
            print(f"row {i}", row_comp)
        comp_lis.append(row_comp)
    print(f"df1 == df2, {all(comp_lis)}")
    return all(comp_lis)


def debug_import_differences():
    """
    Compare performance of models for PEAX, import pipeline on server significantly better performance (fixed)
    Use candy dataset
    :return:
    """
    from .model.ProcessingMethods import NormalizationMethod, DenoisingMethod, ExternalPeakDetectionMethod, PeakDetectionMethod
    # use default values from methods for all of them

    # make analysis
    set_name = "train_full_candy"
    make_plots = False
    plot_parameters, file_parameters = construct_default_parameters(set_name, set_name, make_plots=make_plots,
                                                                    execution_dir_level='project')

    _, evaluation_params_dict = construct_default_processing_evaluation_steps(min_num_cv=3)
    preprocessing_steps = [NormalizationMethod.INTENSITY_NORMALIZATION,
    NormalizationMethod.BASELINE_CORRECTION,
    DenoisingMethod.CROP_INVERSE_REDUCED_MOBILITY,
    DenoisingMethod.DISCRETE_WAVELET_TRANSFORMATION,
    DenoisingMethod.SAVITZKY_GOLAY_FILTER,
    DenoisingMethod.MEDIAN_FILTER,
    DenoisingMethod.GAUSSIAN_FILTER,
    ExternalPeakDetectionMethod.PEAX,
    PeakDetectionMethod.VISUALNOWLAYER,
    PeakAlignmentMethod.PROBE_CLUSTERING,]

    evaluation_params_dict[FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES]['percentage_threshold'] = 0.5  # default

    an = test_resume_analysis(plot_parameters.copy(), file_parameters.copy(), preprocessing_steps,
                                  evaluation_params_dict.copy(), stop_after_alignment=True)
    an.reduce_features(an.AVAILABLE_FEATURE_REDUCTION_METHODS)
    an.evaluate_performance()

    current_pdmn = "PEAX"
    # current_pdmn = "VISUALNOWLAYER"
    # local_fm = an.analysis_result.trainings_matrix['PEAX']  # 33x27
    # server_fm = pd.read_csv("/home/philipp/Desktop/breath_debug/train_PEAX_feature_matrix.csv", index_col=0)  # 33x14
    local_fm = an.analysis_result.trainings_matrix[current_pdmn]  #
    fm = local_fm
    f1 = fm.columns[0]
    sum(fm[f1])
    print(fm)
    sum_up = []
    for col in fm.columns:
        sum_up.append(sum(fm[col]))

    import pickle
    fh = open("/home/philipp/Desktop/breath_debug/fm_peax.p", "rb")
    fm2 = pickle.load(fh)
    fh.close()

    inters_cols = set(fm.columns).intersection(set(fm2.columns))
    sum(fm[:1].values[0])
    sum(fm[inters_cols][:1].values[0])
    sum(fm2[inters_cols][:1].values[0])
    matched_row1 = fm[inters_cols][:1] == fm2[inters_cols][:1]


    server_fm = pd.read_csv(f"/home/philipp/Desktop/breath_debug/train_{current_pdmn}_feature_matrix.csv", index_col=0)

    # both are already reduced - so no difference here
    # server_fm_reduced = AnalysisResult.remove_redundant_features_fm({'PEAX' : server_fm}, an.get_class_label_dict())['PEAX']
    # local_fm_reduced = AnalysisResult.remove_redundant_features_fm({'PEAX' : local_fm}, an.get_class_label_dict())['PEAX']

    local_col_set = set(local_fm.columns.values)
    server_col_set = set(server_fm.columns.values)
    intersecting_features = local_col_set.intersection(server_col_set)

    is_same_list = []
    for f in list(intersecting_features):
        is_same_list.append(local_fm[f] == server_fm[f])
    print(current_pdmn)
    print(sum(is_same_list))
    # is clustering unstable? - how can there be so different results if input is the same? -- sort pdr on input?
    """  PEAX
        BD18_1408280834_ims.csv     6
        BD18_1408280838_ims.csv     8
        BD18_1408280844_ims.csv     2
        BD18_1408280851_ims.csv     7
        BD18_1511121654_ims.csv    27
    """
    """ VISUALNOWLAYER
        BD18_1408280834_ims.csv     0
        BD18_1408280838_ims.csv     0
        BD18_1408280844_ims.csv     0
        BD18_1408280851_ims.csv     0
        BD18_1511121654_ims.csv     0
    """

    # # make sure input is really identical for all measurements [x] - completely identical if sorted
    # pdr_dict = {}
    # for pdmn in ['PEAX', 'VISUALNOWLAYER']:
    #     pdr_comps = []
    #     for i in range(len(an.measurements)):
    #         pdr = an.peak_detection_results[pdmn][i]
    #         df2 = pdr.peak_df
    #         df2.index = df2['measurement_name']
    #         fn1 = f"/home/philipp/Downloads/test/newest/{pdr.measurement_name}_{pdmn}_peak_detection_result.csv"
    #         pdr_comps.append(compare_peax_results(fn1=fn1, fn2="", df2=df2))
    #     pdr_dict[pdmn] = pdr_comps
    # print(pdr_dict)

    pdr_comps = []
    for i in range(fm.shape[0]):
        pdr_df = an.peak_detection_results['PEAX'][i].peak_df
        pdr_df.index = pdr_df['measurement_name']

        fh = open(f"/home/philipp/Desktop/breath_debug/pdr_df_server_peax_{i}.p", "rb")
        pdr_df2 = pickle.load(fh)
        fh.close()
        pdr_df2.index = pdr_df2['measurement_name']
        pdr_comps.append(compare_peax_results(fn1="", fn2="", df1=pdr_df, df2=pdr_df2))

    print(pdr_comps)
    import ipdb; ipdb.set_trace()

def deduplicate_retention_times(retention_times):
    """
    Issue with some old MCC-IMS files (~2006) - retention times are not very consistent - even though it's using floats
        it will write the same retention time twice in seconds, so does some internal rounding ...
    :param retention_times:
    :return:
    """
    retention_times = np.array(retention_times, dtype=float)
    rt_diff = np.array(np.diff(retention_times), dtype=int)
    duplicate_mask = np.logical_not(np.array(rt_diff, dtype=bool))
    duplicate_index = np.nonzero(duplicate_mask)[0][0]  # will get the first index - hopefully we only have to run it once

    # shift until first 2 appears
    # get index of first 2 after duplicate
    # bool(-1) is True - so need to increment to make that work...
    first_2_diff = rt_diff[duplicate_index:]
    first_2_diff[0] = 1

    # might also fail - then we need to shift all behind
    try:
        first2_after_dup_index = np.nonzero(first_2_diff - 1)[0][0]
    except IndexError:
        first2_after_dup_index = len(first_2_diff)

    first2_index = duplicate_index + first2_after_dup_index + 1  # need to do +1 due to index shift by np.diff

    # including first duplicate
    part1 = retention_times[:duplicate_index + 1]
    part2 = retention_times[duplicate_index + 1:first2_index] + 1
    part3 = retention_times[first2_index:]

    # deduplicated list
    return np.concatenate([part1, part2, part3])  # array is ok for later usage