"""
nexus/brand.py
==============
NEXUS brand identity — ASCII art, sigils, and color constants.

Project story:
    In StarCraft, the Nexus is the psionic heart of every Protoss colony.
    It is the first structure to warp in — the point from which probes
    emerge to harvest scattered fields of minerals and vespene, and the
    gate through which all resources, and all will, return.

    The Nexus is convergence made physical. Without it, probes drift
    alone. Expeditions fragment into isolated efforts, and the colony's
    shared purpose is lost.

    This project carries the same mandate.

    Dozens of RL experiments scatter across GPU servers — each one a
    probe harvesting its own insight: reward curves, gradient norms,
    success rates, tactile traces. Left alone, these insights sit in
    the dark: tfevents on personal machines, metrics trapped in isolated
    training boxes, runs nobody else on the team can see or search.

    NEXUS is the central warp-gate.
    Every run reports back. Every hyperparameter is preserved. Every
    learning curve joins the others, ready to be compared, searched,
    and reasoned over as one.

    The scattered becomes whole. The many become one.

    En Taro Adun.

Usage:
    from brand import print_banner, SIGIL

    print_banner()
    print(f"{SIGIL} Upload complete")
"""

# ── ANSI color codes ──────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
BLUE = "\033[94m"
WHITE = "\033[97m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
PURPLE = "\033[95m"
RED = "\033[91m"

# ── Inline sigil — use in log lines and CLI prompts ───────────────────
#
#   [NXS]  (bold cyan)
#
SIGIL = f"{CYAN}{BOLD}[NXS]{RESET}"

# ── Full startup banner ───────────────────────────────────────────────
#
#  Visual language:
#    · Outer psi field   — psionic energy radiating from the Nexus
#    · Angular frame     — Protoss crystalline geometry (◆ corners, ━ ┃ edges)
#    · Central crystal   — ⬢ the psionic core of the colony
#    · Probe convergence — scattered runs warp back to the Nexus
#
#  Harvest flow diagram:
#
#    ●   ●   ●   ●   ●   ← experiment runs (scattered)
#    │   │   │   │   │
#    └───┴───┼───┴───┘   ← logs warped to the Nexus
#            │
#          NEXUS         ← compare · search · deploy
#
BANNER = f"""{CYAN}{BOLD}
  {RESET}{CYAN}◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆{BOLD}
  ┃                                            ┃
  ┃             {WHITE}{BOLD}⬢   N E X U S   ⬢             {CYAN}{BOLD} ┃
  ┃                                            ┃
  ┃  {RESET}{YELLOW}All runs warped home. No log left behind.{CYAN}{BOLD} ┃
  ┃                                            ┃
  {RESET}{CYAN}◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆{BOLD}
{RESET}"""

# ── Probe convergence flow diagram (standalone) ──────────────────────
FLOW = (
    f"{DIM}{CYAN}  ●   ●   ●   ●   ●{RESET}  "
    f"{DIM}← experiment runs (scattered){RESET}\n"
    f"{CYAN}  │   │   │   │   │{RESET}\n"
    f"{CYAN}  └───┴───┼───┴───┘{RESET}  "
    f"{DIM}← logs warped to the Nexus{RESET}\n"
    f"{CYAN}          │{RESET}\n"
    f"        {WHITE}{BOLD}NEXUS{RESET}        "
    f"{DIM}← compare · search · deploy{RESET}\n"
)

# ── Version ───────────────────────────────────────────────────────────
VERSION = "0.1.0"
VERSION_STRING = f"{CYAN}{BOLD}NEXUS{RESET} {DIM}v{VERSION}{RESET}"


# ── Public functions ──────────────────────────────────────────────────


def print_banner() -> None:
    """Print the full NEXUS startup banner."""
    print(BANNER)


def print_flow() -> None:
    """Print the probe convergence flow diagram."""
    print(FLOW)


def rule(title: str = "", width: int = 54) -> str:
    """Return a styled horizontal rule with an optional centered title."""
    if title:
        pad = (width - len(title) - 2) // 2
        line = f"{'─' * pad} {title} {'─' * (width - len(title) - 2 - pad)}"
    else:
        line = "─" * width
    return f"{CYAN}{line}{RESET}"


def log(msg: str, level: str = "info") -> str:
    """
    Return a formatted log prefix line.

    Parameters
    ----------
    msg   : message text
    level : "info" | "ok" | "warn" | "error"
    """
    icons = {
        "info": f"{CYAN}[NXS]{RESET}",
        "ok": f"{GREEN}[NXS]{RESET}",
        "warn": f"{YELLOW}[NXS]{RESET}",
        "error": f"{RED}[NXS]{RESET}",
    }
    prefix = icons.get(level, icons["info"])
    return f"{prefix} {msg}"


if __name__ == "__main__":
    print_banner()
    print()
    print_flow()
    print()
    print(f"  Sigil   : {SIGIL}")
    print(f"  Version : {VERSION_STRING}")
    print()
    print(rule("Warp Complete"))
    print()
    print(log("5 runs synced to NEXUS", "info"))
    print(log("Experiment 'robot_hand_rl' indexed", "ok"))
    print(log("2 runs missing required tags", "warn"))
    print(log("MLflow server unreachable (connection refused)", "error"))
