# Streamlit Community Cloud deployment

Deploy the verified release branch with these settings:

- Repository: `Remigaraki/TwoStream-Audio-Forensics`
- Branch: `codex/streamlit-demo`
- Main file path: `demo/app.py`
- Advanced settings → Python: `3.11`
- Secrets: none required for the public, Git-LFS-backed model assets.

Community Cloud checks dependency files beside the entrypoint before the root;
`demo/requirements.txt` pins the verified CPU runtime. The root
`.streamlit/config.toml` sets the upload limit and visitor error behavior.
Git LFS is supported by Community Cloud. Both served checkpoints and the PCA
are checked by SHA-256 before loading. Thresholds and parity reports must ship
together with the corrected source.

Before publication, run `demo/.venv/Scripts/python demo/verify_release.py` and
the local demo tests. After deployment, check build logs, open the app and test
a supported speech recording, both model results, short audio, silence and a
corrupt file. Confirm threshold values C1 `0.5611` and A1 `0.9232`.

If a dependency, identity, or LFS check fails, keep the guard enabled and fix the
release; do not replace weights, remove hash checks, or silently use defaults.
No training dataset or GitHub token belongs in the hosting app.

References:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
