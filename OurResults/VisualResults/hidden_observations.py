""" File with the observations generated manually when analyzing the layer types
present at the most influential positions of the DNN encoding as described by 
permutation importance."""

{
    "2013": [
        ["TanH", "TanH", "Identity", "Linear(1)", "Conv1D(2, 2)"],
        "4 classes of layers, all except Dropout."
        ["Linear(1)", "Linear(8)", "Linear(32)", "Linear(4)", "Linear(16)"],
        "Linear with few neurons (1, 8, 32, 4, etc.) appears 5 times."
        ["Dropout(0.6)", "Linear(256)", "Dropout(0.6)", "Linear(1)", "Conv1D(2, 8)"],
        "Dropout and Linear layers appear twice."
        ["Conv1D(8, 8)", "Linear(1024)", "ReLU", "Conv1D(8, 2)", "Linear(64)"],
        "Conv1D and Linear layers appear twice."
    ],
    "2014Essay2": [
        ["TanH", "ReLU", "Linear(256)", "Conv1D(8, 2)", "Linear(32)"],
        "Activation and Linear layers appear twice."
        ["Linear(512)", "Linear(512)", "Linear(4096)", "Linear(8)", "Linear(512)"],
        "Linear with moderate/many neurons (512, 4096) appears 4 times."
        ["Linear(8)", "Linear(8)", "TanH", "Linear(8)", "Linear(8)"],
        "Linear with few neurons (8) appears 4 times."
        ["Linear(1)", "Linear(2)", "Linear(1)", "Linear(1)", "Linear(16)"],
        "Linear with few neurons (1, 2, 16, etc.) appears 5 times."
    ],
    "2014Novel2": [
        ["Linear(4096)", "Linear(4096)", "Linear(32)", "Linear(4096)", "Linear(4096)"],
        "Linear with many neurons (4096) appears 4 times."
        ["Linear(32)", "Linear(4096)", "Linear(2048)", "Linear(2)", "Linear(32)"],
        "Linear with any neurons (32, 4096, 2, etc.) appears 5 times."
        ["Linear(64)", "Identity", "ReLU", "Linear(256)", "ReLU"],
        "Activation and Linear layers appear twice."
        ["Linear(256)", "Dropout(0.6)", "Linear(64)", "Conv1D(2, 8)", "Linear(256)"],
        "Linear with moderate neurons (256, 64) appears 3 times."
    ],
    "2015": [
        ["Linear(4)", "ReLU", "ReLU", "Linear(2048)", "Linear(256)"],
        "Linear with any neurons (4, 2048, 256) appears 3 times, activation appears twice."
        ["Linear(4)", "Leaky ReLU", "Linear(4)", "ReLU", "Linear(4)"],
        "Linear with few neurons (4) appears 3 times, activation appears twice."
        ["Linear(64)", "Dropout(0.9)", "Linear(2)", "Linear(2)", "Linear(16)"],
        "Linear with few/moderate neurons (2, 2, 16, 64) appears 4 times."
        ["Linear(64)", "Linear(64)", "Conv1D(2, 8)", "Linear(64)", "Linear(1024)"],
        "Linear with moderate/many neurons (64, 1024) appears 3 times."
    ],
    "2020": [
        ["Linear(1)", "Linear(16)", "ReLU", "Linear(16)", "Linear(64)"],
        "Linear with few/moderate neurons (1, 16, 64) appears 4 times."
        ["Linear(256)", "Identity", "Conv1D(8, 8)", "Identity", "Linear(256)"],
        "Identity and Linear layers appear twice."
        ["TanH", "Linear(4096)", "TanH", "Linear(256)", "Linear(1024)"],
        "Linear with moderate/many neurons (256, 1024, 4096) appears 3 times, activation appears twice."
        ["Linear(16)", "Linear(64)", "Linear(512)", "Linear(256)", "TanH"],
        "Linear with few/moderate neurons (16, 64, 256, 512) appears 4 times."
    ],
}
