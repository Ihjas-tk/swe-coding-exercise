#!/usr/bin/env bash
# One-command setup: system packages, pinned Python environment, .env, model cache, `ae check`.
#
#   ./setup.sh                 # macOS (Homebrew) or Debian/Ubuntu (apt)
#   SKIP_LIBREOFFICE=1 ./setup.sh   # do not install LibreOffice (DOCX pages then default to 1)
#   SKIP_MODELS=1 ./setup.sh        # do not pre-download models (they download on first ingest)
#
# Idempotent: every step checks first and only installs what is missing, so re-running is safe.
#
# Expected runtime (M-series laptop, good network):
#   everything already present ............................. ~1 min (uv sync + model check + ae check)
#   uv + Python deps from scratch .......................... +1-2 min (~1.5 GB of wheels incl. torch)
#   Tesseract via Homebrew / apt ........................... +1-3 min
#   LibreOffice (Homebrew cask ~700 MB / apt ~300 MB) ...... +2-5 min (optional)
#   models (embedding ~290 MB, Docling ~500 MB) ............ +1-3 min
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
START=$(date +%s)

log()  { printf '\n[%s] ==> %s\n' "$(date +%H:%M:%S)" "$*"; }
info() { printf '[%s]     %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[%s] !!  %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '[%s] xx  %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------------------
log "1/7 Detecting platform"
OS=unsupported
case "$(uname -s)" in
  Darwin) OS=macos ;;
  Linux)
    if [ -r /etc/os-release ]; then
      # shellcheck disable=SC1091
      . /etc/os-release
      case " ${ID:-} ${ID_LIKE:-} " in
        *" debian "* | *" ubuntu "*) OS=debian ;;
      esac
    fi
    ;;
esac
info "platform: $OS ($(uname -sm))"

SUDO=""
if [ "$OS" = debian ] && [ "$(id -u)" -ne 0 ]; then
  have sudo || die "not root and sudo is not installed; re-run as root or install sudo"
  SUDO="sudo"
fi

APT_UPDATED=0
apt_install() {
  if [ "$APT_UPDATED" -eq 0 ]; then
    info "apt-get update"
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq
    APT_UPDATED=1
  fi
  info "apt-get install $*"
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$@"
}

need_brew() {
  have brew && return 0
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
    if [ -x "$b" ]; then eval "$("$b" shellenv)"; return 0; fi
  done
  return 1
}

if [ "$OS" = unsupported ]; then
  warn "only macOS and Debian/Ubuntu are automated; install uv, Tesseract 5 (+ English data) and"
  warn "optionally LibreOffice yourself; the remaining steps run if uv is on PATH."
fi

# ---------------------------------------------------------------------------------------
log "2/7 uv (Python package and interpreter manager)"
if ! have uv && [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi
if ! have uv && [ -x "$HOME/.cargo/bin/uv" ]; then export PATH="$HOME/.cargo/bin:$PATH"; fi
if have uv; then
  info "found $(uv --version) at $(command -v uv)"
else
  if ! have curl; then
    if [ "$OS" = debian ]; then apt_install --no-install-recommends curl ca-certificates; else die "curl is required to install uv"; fi
  fi
  info "installing uv with the official installer (https://astral.sh/uv/install.sh)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  have uv || die "uv installed but not on PATH; open a new shell (or add ~/.local/bin to PATH) and re-run"
  info "installed $(uv --version)"
fi

# ---------------------------------------------------------------------------------------
log "3/7 Tesseract OCR with English data (required: scanned pages and figure labels)"
tesseract_ok() {
  have tesseract || return 1
  local langs
  langs=$(tesseract --list-langs 2>/dev/null) || return 1
  grep -qx eng <<<"$langs"
}
tesseract_version() { local v; v=$(tesseract --version 2>&1) || true; printf '%s\n' "${v%%$'\n'*}"; }
if tesseract_ok; then
  info "found $(tesseract_version) with 'eng'"
else
  case "$OS" in
    macos)
      need_brew || die "Homebrew not found: install it from https://brew.sh (or install tesseract yourself) and re-run"
      brew install tesseract
      ;;
    debian) apt_install --no-install-recommends tesseract-ocr tesseract-ocr-eng ;;
    *) warn "install Tesseract 5 with English data manually" ;;
  esac
  if tesseract_ok; then
    info "installed $(tesseract_version)"
  elif [ "$OS" != unsupported ]; then
    die "tesseract is installed but the 'eng' language data is missing ($(command -v tesseract || echo 'not on PATH'))"
  fi
fi

# ---------------------------------------------------------------------------------------
log "4/7 LibreOffice (optional: page numbers for DOCX files)"
soffice_path() {
  for c in soffice libreoffice; do have "$c" && { command -v "$c"; return 0; }; done
  [ -x /Applications/LibreOffice.app/Contents/MacOS/soffice ] && { echo /Applications/LibreOffice.app/Contents/MacOS/soffice; return 0; }
  return 1
}
LO_MISSING_MSG="LibreOffice not installed: DOCX blocks will be reported on page 1 (everything else works)"
if p=$(soffice_path); then
  info "found $p"
elif [ "${SKIP_LIBREOFFICE:-0}" = 1 ]; then
  warn "SKIP_LIBREOFFICE=1: $LO_MISSING_MSG"
else
  ok=0
  case "$OS" in
    macos)
      if need_brew; then
        info "brew install --cask libreoffice (~700 MB; set SKIP_LIBREOFFICE=1 to skip)"
        brew install --cask libreoffice && ok=1
      fi
      ;;
    debian) apt_install libreoffice-writer && ok=1 ;;  # with recommends: fonts affect pagination
  esac
  if [ "$ok" = 1 ] && p=$(soffice_path); then info "installed $p"; else warn "$LO_MISSING_MSG"; fi
fi

# ---------------------------------------------------------------------------------------
log "5/7 Python environment from uv.lock (Python 3.12 is fetched by uv if missing)"
uv sync --locked
info "environment ready at $(pwd)/.venv"

# ---------------------------------------------------------------------------------------
log "6/7 .env (ANTHROPIC_API_KEY)"
KEY_RE='^[[:space:]]*ANTHROPIC_API_KEY[[:space:]]*=[[:space:]]*[^[:space:]#]'
# ae reads only the nearest .env (this directory, then each parent), like ae/config.py.
nearest_dotenv() {
  local d
  d=$(pwd)
  while :; do
    [ -f "$d/.env" ] && { echo "$d/.env"; return 0; }
    [ "$d" = / ] && return 1
    d=$(dirname "$d")
  done
}
if [ ! -f .env ] && p=$(nearest_dotenv) && grep -Eq "$KEY_RE" "$p"; then
  # a repo-level .env would shadow it, so do not create one
  info "using $p, which already sets ANTHROPIC_API_KEY (no .env created here)"
else
  if [ ! -f .env ]; then
    cp .env.example .env
    chmod 600 .env
    info "created .env from .env.example"
  else
    info ".env already exists; left unchanged"
  fi
  if grep -Eq "$KEY_RE" .env; then
    info "ANTHROPIC_API_KEY is set in .env"
  elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    info "ANTHROPIC_API_KEY is set in the environment (it takes precedence over .env)"
  elif [ -t 0 ]; then
    printf 'Paste your ANTHROPIC_API_KEY (input hidden; press Enter to skip): '
    IFS= read -r -s key || key=""
    printf '\n'
    if [ -n "$key" ]; then
      tmp=$(mktemp)
      grep -Ev '^[[:space:]]*ANTHROPIC_API_KEY[[:space:]]*=' .env > "$tmp" || true
      printf 'ANTHROPIC_API_KEY=%s\n' "$key" >> "$tmp"
      mv "$tmp" .env
      chmod 600 .env
      unset key
      info "saved ANTHROPIC_API_KEY to .env"
    else
      warn "no key entered: edit .env and set ANTHROPIC_API_KEY=... before make smoke / ask / eval"
    fi
  else
    warn "ANTHROPIC_API_KEY is empty: edit $(pwd)/.env and set ANTHROPIC_API_KEY=..."
    warn "(make ingest works without it; make smoke / ask / eval need it)"
  fi
fi

# ---------------------------------------------------------------------------------------
log "7/7 Models (Hugging Face cache) and environment check"
if [ "${SKIP_MODELS:-0}" = 1 ]; then
  warn "SKIP_MODELS=1: models download on first ingest instead (~790 MB)"
else
  info "embedding model + Docling layout/TableFormer (~790 MB on first run)"
  if ! uv run python -c "from ae import config; from ae.models import prefetch; prefetch(config.EMBED_MODEL)"; then
    warn "model pre-download failed; they will download on first ingest (see README > Troubleshooting)"
  fi
fi

CHECK_RC=0
uv run ae check || CHECK_RC=$?

ELAPSED=$(( $(date +%s) - START ))
log "Done in $((ELAPSED / 60)) min $((ELAPSED % 60)) s"
if [ "$CHECK_RC" -ne 0 ]; then
  warn "ae check reported failures (lines marked ✗ above); fix them before ingesting"
fi
cat <<'EOF'

Next steps:
  1. Make sure ANTHROPIC_API_KEY is set in .env
  2. make ingest                # ~2 min: index the 9 files in patents/ design_docs/ structured/
  3. make ask Q="What is the maximum discharge current rating of the EV-BMS-100?"
  4. make eval                  # ~3 min: staged evaluation -> data/eval/RESULTS.md
  Optional: make smoke          # ~4 min self-test on one file of each type, in its own index
  make help lists every target.
EOF
exit "$CHECK_RC"
