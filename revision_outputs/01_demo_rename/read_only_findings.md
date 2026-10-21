Not implemented: user requested information/data only. Current labels requiring the proposed rename:

demo/app.py:20:     "Its output is a model score, not proof: it does not establish a speaker's identity "
demo/app.py:41:     c1.metric("Model score", f"{s:.4f}", help="Sigmoid output of the network; higher = more spoof-like. "
demo/README.md:3: Upload a speech recording and get a **C1** model score and a bona fide / spoofed
demo/README.md:47: `Sigmoid`, and the demo shows that output as the **model score** with no second sigmoid. It is not
demo/space/README.md:17: This is an experimental research model. Its output is a model score, not proof of
demo/tests/test_demo.py:201:     assert [m.label for m in at.metric].count("Model score") == 2

Use "spoof score" for the continuous sigmoid output; higher means more spoof-like. The binary class decision is a separate thresholded output. Calibration was not evaluated; do not describe the score as calibrated probability. No tests, branch, commit, push or deployment were performed.
