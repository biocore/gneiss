from ete3 import faces, AttrFace, CircleFace, BarChartFace


def default_layout(node):
    """
    Specifies the layout for the ete.TreeStyle object.

    Parameters
    ----------
    node: ete.Tree
        Input node for specifying which attributes.
    """
    if node.is_leaf():
        # Add node name to leaf nodes
        N = AttrFace("name", fsize=14, fgcolor="black")

        faces.add_face_to_node(N, node, 0)
    if "weight" in node.features:
        # Creates a sphere face whose size is proportional to node's
        # feature "weight"
        C = CircleFace(radius=node.weight, color="Red", style="sphere")
        # Let's make the sphere transparent
        C.opacity = 0.5
        # Rotate the faces by 90*
        C.rotation = 90
        # And place as a float face over the tree
        faces.add_face_to_node(C, node, 0, position="float")


def barchart_layout(node, name='name',
                    width=20, height=40,
                    colors=None, min_value=0, max_value=1,
                    fsize=14, fgcolor="black",
                    alpha=0.5,
                    rotation=270):
    """
    Specifies the layout for the ete.TreeStyle object.

    Parameters
    ----------
    node: ete.Tree
        Input node for specifying which attributes.
    name: str, optional
        Attribute to look up the name of the node.
    width: int, optional
        Width of the barchart.
    height: int, optional
        Height of the barchart.
    colors: list of str, optional
        List of HTML colors to color the barchart values.
    min_value: int, optional
        Minimum value to set the scale of the chart.
    max_value: int, optional
        Maximum value to set the scale of the chart.
    fsize: int, optional
        Font size on the leafs.
    fgcolor: str, optional
        Font color of the leafs.
    alpha: float, optional
        Transparency of the barchart.
    rotation: int, optional
        Orientation of the barchart.
    """
    if colors is None:
        colors = ['#0000FF']
    if node.is_leaf():
        # Add node name to leaf nodes
        N = AttrFace("name", fsize=fsize, fgcolor=fgcolor)
        faces.add_face_to_node(N, node, 0)
    if "weight" in node.features:
        # Creates a sphere face whose size is proportional to node's
        # feature "weight"
        if (isinstance(node.weight, int) or isinstance(node.weight, float)):
            weight = [node.weight]
        else:
            weight = node.weight
        C = BarChartFace(values=weight, width=width, height=height,
                         colors=colors, min_value=min_value,
                         max_value=max_value)
        # Let's make the sphere transparent
        C.opacity = alpha
        # Rotate the faces by 270*
        C.rotation = rotation
        # And place as a float face over the tree
        faces.add_face_to_node(C, node, 0, position="float")
