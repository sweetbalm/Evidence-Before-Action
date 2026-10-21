import xml.etree.ElementTree as ET
from typing import Tuple


def find_leaf_nodes(xlm_file_str):
    if not xlm_file_str:
        return []

    root = ET.fromstring(xlm_file_str)

    # Recursive function to traverse the XML tree and collect leaf nodes
    def collect_leaf_nodes(node, leaf_nodes):
        # If the node has no children, it is a leaf node, add it to the list
        if not list(node):
            leaf_nodes.append(node)
        # If the node has children, recurse on each child
        for child in node:
            collect_leaf_nodes(child, leaf_nodes)

    # List to hold all leaf nodes
    leaf_nodes = []
    collect_leaf_nodes(root, leaf_nodes)
    return leaf_nodes


state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"


def judge_node(node: ET, platform="ubuntu") -> bool:
    if platform == "ubuntu":
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
    elif platform == "windows":
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    keeps: bool = node.tag.startswith("document") \
                  or node.tag.endswith("item") \
                  or node.tag.endswith("button") \
                  or node.tag.endswith("heading") \
                  or node.tag.endswith("label") \
                  or node.tag.endswith("scrollbar") \
                  or node.tag.endswith("searchbox") \
                  or node.tag.endswith("textbox") \
                  or node.tag.endswith("link") \
                  or node.tag.endswith("tabelement") \
                  or node.tag.endswith("textfield") \
                  or node.tag.endswith("textarea") \
                  or node.tag.endswith("menu") \
                  or node.tag in {"alert", "canvas", "check-box"
                      , "combo-box", "entry", "icon"
                      , "image", "paragraph", "scroll-bar"
                      , "section", "slider", "static"
                      , "table-cell", "terminal", "text"
                      , "netuiribbontab", "start", "trayclockwclass"
                      , "traydummysearchcontrol", "uiimage", "uiproperty"
                      , "uiribboncommandbar"
                                  }
    keeps = keeps and (
            platform == "ubuntu"
            and node.get("{{{:}}}showing".format(_state_ns), "false") == "true"
            and node.get("{{{:}}}visible".format(_state_ns), "false") == "true"
            or platform == "windows"
            and node.get("{{{:}}}visible".format(_state_ns), "false") == "true"
    ) \
            and (
                    node.get("{{{:}}}enabled".format(_state_ns), "false") == "true"
                    or node.get("{{{:}}}editable".format(_state_ns), "false") == "true"
                    or node.get("{{{:}}}expandable".format(_state_ns), "false") == "true"
                    or node.get("{{{:}}}checkable".format(_state_ns), "false") == "true"
            ) \
            and (
                    node.get("name", "") != "" or node.text is not None and len(node.text) > 0
            )

    coordinates: Tuple[int, int] = eval(node.get("{{{:}}}screencoord".format(_component_ns), "(-1, -1)"))
    sizes: Tuple[int, int] = eval(node.get("{{{:}}}size".format(_component_ns), "(-1, -1)"))
    keeps = keeps and coordinates[0] >= 0 and coordinates[1] >= 0 and sizes[0] > 0 and sizes[1] > 0
    return keeps


def filter_nodes(root: ET, platform="ubuntu"):
    filtered_nodes = []

    for node in root.iter():
        if judge_node(node, platform):
            filtered_nodes.append(node)
            # print(ET.tostring(node, encoding="unicode"))

    return filtered_nodes


def print_nodes_with_indent(nodes, indent=0):
    for node in nodes:
        print(' ' * indent, node.tag, node.attrib)
        print_nodes_with_indent(node, indent + 2)
