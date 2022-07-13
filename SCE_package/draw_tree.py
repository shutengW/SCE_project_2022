from pathlib import Path

# prefix components:
space = '    '
branch = '│   '
# pointers:
tee = '├── '
last = '└── '


def tree(dir_path: Path, prefix: str = ''):
    """A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters

    Args:
        dir_path (Path): Path of the directory to be traversed
        prefix (str, optional): gPrefix of the drawin. Defaults to ''.

    Yields:
        tree: Tree structure line by line describing the structure of the Python package.
    """

    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir():  # extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix+extension)


for line in tree(Path.home() / 'Desktop' / 'RWTH 1semester' / 'RWTH 4th semester' / 'SCE' / 'SCE_package'):
    print(line)
