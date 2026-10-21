The manifest builder fixes TARGET_SPOOF_PER_ATTACK=4687 (scripts/build_manifest.py:40). It retains all bona fide and samples each attack evenly (lines 106-120). This proves the construction, not why the cap was selected.

- Arithmetic-only inference, not stated by Rafa: floor(75,000 / 16) = 4,687. A target of 75,000 spoofed is compatible, but unconfirmed. Adding all 50,131 bona fide to the unrounded target would give 125,131; the rounded cap gives 125,123. [computed arithmetic]
- Arithmetic-only inference: (125,000 - 50,131) / 16 = 4,679.3125, not 4,679.4 as rounded in the megaprompt. A direct equal allocation under a strict total target would not produce 4,687. [computed arithmetic]
- Smallest available attack hypothesis: NOT FOUND; source train/dev metadata absent. The returned manifest confirms 4,687 sampled per attack but cannot recover unsampled availability.
- Time/storage rationale and intention to rebalance: NOT FOUND; pending Rafa.

Original pool class fractions using the megaprompt pool count: bona fide 15.5057%, spoofed 84.4943% [computed from megaprompt: 50,131 / 323,307]. Subsample bona fide 40.0654%, spoofed 59.9346% [computed from manifest]. Reduced imbalance is observed; intent and sufficiency to dispense with weighting are not established. src/train.py:263 uses shuffle=True; line 328 uses nn.BCELoss() without weights; lines 262-265 pass no weighted sampler. This is current code, not proof every historical run used this revision.

Rafa supplied an evidence-review account, preserved verbatim in rafa_answers.md. It does not provide firsthand confirmation of the exact cap rationale, compute budget or prespecification date.
