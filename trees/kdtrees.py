class Node:
    cut_val = None
    min_val = None
    max_val = None

    def __init__(self, parent, left=None, right=None):
        self.parent  = parent

    def __repr__(self):
        return "Node(None, {}, {}, {})".format(self.cut_val,
                                               self.min_val, self.max_val)


def generate_tree(min_val, max_val, levels, parent=None):
    if levels == 0:
        return []
    else:
        mid_val = (min_val + max_val) / 2
        node = Node(parent, mid_val)
        node.cut_val = mid_val

        return  ( generate_tree(min_val, mid_val, levels-1, node)
                + [node]
                + generate_tree(mid_val, max_val, levels-1, node))


def set_min_max(tree):
    for node in tree:
        parent = node.parent

        while parent is not None:
            if parent.cut_val > node.cut_val:
                if node.max_val is not None:
                    node.max_val = min(node.max_val, parent.cut_val)
                else:
                    node.max_val = parent.cut_val
            else:
                if node.min_val is not None:
                    node.min_val = max(node.min_val, parent.cut_val)
                else:
                    node.min_val = parent.cut_val

            parent = parent.parent


tree = generate_tree(0.0, 10.0, 3)
set_min_max(tree)

print len(tree), tree
