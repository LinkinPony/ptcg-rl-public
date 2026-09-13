# Scope of the MIT license

The MIT license covers the original project code and our trained model weights.
It does not grant rights to the Pokemon characters, card names, artwork, game
content, trademarks, competition data, or third-party software.

The competition simulator is **not open source**. Its supplied license expressly
prohibits public redistribution. This repository excludes the engine sources,
the `cg` SDK, and native libraries compiled with the engine. It also excludes
vendored competitors' agents whose redistribution permissions were not established.
Refer to the [competition rules](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules)
and obtain any required authorization independently. Possession of an old SDK copy
does not itself establish a continuing right to use or redistribute it.

The two submission archives have therefore been sanitized. Their learned weights,
decklists, inference temperature, and inference catalog are byte-identical to the
corresponding original submissions. Decklists and game-related catalog data remain
subject to the rights of their respective owners; they are not relicensed by MIT.
The archives omit the engine, unrelated frontend assets, vendored opponent agents,
and internal provenance records. Machine-specific remote deployment defaults were
removed from two Python modules. Archive ownership and timestamps were normalized.
Each archive includes an exact list of removed, modified, and unchanged members.

Third-party packages installed through Python or npm retain their own licenses.
