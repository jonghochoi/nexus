# ============================================================
# scheduled_sync/_activate_venv.sh
#
# Shared venv activation — source this file from any sibling script.
# Prefers the shared ~/.nexus/venv (created by setup.sh) and falls back
# to a repo-local ./venv for legacy installs. A no-op when neither
# exists, so the caller can still run if Python tooling is on PATH.
#
# Usage (from any scheduled_sync/*.sh):
#   # shellcheck source=_activate_venv.sh disable=SC1091
#   source "$(dirname "${BASH_SOURCE[0]}")/_activate_venv.sh"
# ============================================================

if [ -f "${HOME}/.nexus/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/.nexus/venv/bin/activate"
elif [ -f "venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi
