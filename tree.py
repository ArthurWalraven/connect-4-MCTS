import pickle


class Node():

    def __init__(self, parent, data = None):

        self.parent = parent
        self.data = data
        self.children = []
    

    def __str__(self):
        
        return str(self.data)



def count_decendants(node : Node) -> int:

    if (not node.children):

        return 0
    

    ans = len(node.children)

    for c in node.children:

        ans += count_decendants(c)
    
    
    return ans


def add_child(node : Node, child_data = None) -> Node:

    node.children.append(Node(parent=node, data=child_data))
    return node.children[-1]


def path_to_child(root : Node, target_child : Node) -> list:
    
    path = []

    _path_to_child(root, target_child, path)

    return path


def console_render(node : Node, highlight_child : Node = None, file = None) -> None:

    _console_render(
        node,
        '',
        '',
        path_to_child(node, highlight_child) if highlight_child else [],
        file
    )


def _path_to_child(node : Node, target_child : Node, path_to_target : list) -> bool:

    if (id(node) == id(target_child)):

        return True
    
    for i in range(len(node.children)):

        if (_path_to_child(node.children[i], target_child, path_to_target)):
            
            path_to_target.insert(0, i)

            return True


def _console_render(node : Node, indentation: str, tree_char: str, highlight_path: list, file = None) -> None:

    char_draw_lines = str(node).split('\n')


    if (char_draw_lines):

        print(indentation[:-len(tree_char)] + tree_char + char_draw_lines[0], file=file)
        
        for char_line in char_draw_lines[1:]:

            print(indentation + char_line, file=file)
    

    if (not node.children):

        return


    highlighted_index = -1

    if (highlight_path):

        highlighted_index = highlight_path[0]

        for child in node.children[:highlighted_index]:

            # print(indentation + '║')
            _console_render(child, indentation + '║  ', '╟─ ', [], file)


        if (highlighted_index < (len(node.children) - 1)):

            # print(indentation + '║')
            _console_render(node.children[highlighted_index], indentation + '│  ', '╠═ ', highlight_path[1:], file)
        
        else:

            # print(indentation + '║')
            _console_render(node.children[-1], indentation + '   ', '╚═ ', highlight_path[1:], file)

    for child in node.children[highlighted_index + 1 : -1]:

        # print(indentation + '│')
        _console_render(child, indentation + '│  ', '├─ ', [], file)

    if (highlighted_index < (len(node.children) - 1)):

        # print(indentation + '│')
        _console_render(node.children[-1], indentation + '   ', '└─ ', [], file)


def pickle_tree(root : Node, file_name : str) -> None:

    def pickle_node(node : Node, pickle_file) -> None:

        pickle.dump([c.data for c in node.children], pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        for c in node.children:

            pickle_node(c, pickle_file)


    with open(file_name, 'wb') as bin_file:

        pickle.dump(root.data, bin_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_node(root, bin_file)


def unpickle_tree(file_name : str) -> Node:

    def unpickle_node(node : Node, pickle_file) -> None:

        node.children = [Node(parent=node, data=d) for d in pickle.load(pickle_file)]

        for c in node.children:

            unpickle_node(c, pickle_file)
    

    with open(file_name, 'rb') as bin_file:

        root = Node(parent=None, data=pickle.load(bin_file))
        
        unpickle_node(root, bin_file)

        return root


if __name__ == "__main__":

    class ComplexData():

        def __init__(self, s : str):

            self.s = s
            self.chr_list = [c for c in s]
        
        def __str__(self):

            return self.s + ' ' + str(self.chr_list)
    

    root = Node(None, ComplexData('Root'))

    a = add_child(root, ComplexData('a'))
    b = add_child(root, ComplexData('b'))

    aa = add_child(a, ComplexData('aa'))
    ab = add_child(a, ComplexData('ab'))

    ba = add_child(b, ComplexData('ba'))
    bb = add_child(b, ComplexData('bb'))

    aba = add_child(ab, ComplexData('aba'))
    abb = add_child(ab, ComplexData('abb'))

    baa = add_child(ba, ComplexData('baa'))

    abaa = add_child(aba, ComplexData('abaa'))
    abab = add_child(aba, ComplexData('abab'))
    abac = add_child(aba, ComplexData('abac'))


    console_render(root, highlight_child=abab)
    print('Path to highlighted child:', (path_to_child(root, abab)))

    # Save to file
    pickle_tree(root=root, file_name='delete_me')

    # Load from file
    root2 = unpickle_tree('delete_me')

    # The tree structure is saved
    console_render(root2, highlight_child=abab)

    # But the loaded node are different entities: no 'abab' in the loaded tree
    print('Path to highlighted child:', (path_to_child(root2, abab)))
    