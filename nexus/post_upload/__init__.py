"""nexus.post_upload — Pipeline B uploader & history utilities.

Importable surface (used by external glue scripts):

    from nexus.post_upload.upload_eval import upload_eval
    from nexus.post_upload.run_info  import read_run_info  # re-exported via nexus.logger

CLI entry points are exposed through ``[project.scripts]`` in pyproject.toml
(``nexus-upload-tb``, ``nexus-upload-eval``, ``nexus-verify-tb``); equivalent
``python -m nexus.post_upload.<module>`` invocations also work.
"""
