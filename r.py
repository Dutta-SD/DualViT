import newick

with open("data/imagenet_meta/ref.nwk", "r") as f:
    root = newick.load(f)[0]
    print(root.ascii_art())