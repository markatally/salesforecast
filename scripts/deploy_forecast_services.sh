#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sudo bash scripts/deploy_forecast_services.sh --account USER:PASS [--account USER2:PASS2 ...] [options]

Installs and configures:
  - Git
  - Miniforge3
  - JupyterHub + JupyterLab
  - Streamlit
  - File Browser

Options:
  --account USER:PASS          Linux/JupyterHub user and File Browser admin.
                               May be repeated.
  --miniforge-dir DIR          Miniforge install path. Default: /opt/miniforge3
  --share-dir DIR              File Browser public root. Default: /srv/team-share
  --streamlit-app PATH         Streamlit app file to run.
                               Default: Streamlit built-in hello app
  --jupyterhub-port PORT       Default: 8000
  --streamlit-port PORT        Default: 8501
  --filebrowser-port PORT      Default: 8080
  --proxy URL                  Optional proxy for curl/apt/conda, e.g.
                               socks5h://127.0.0.1:1080
  --sudo-password PASS         Optional sudo password when not running as root.
  -h, --help                   Show this help.

Examples:
  sudo bash scripts/deploy_forecast_services.sh \
    --account mark:881018 \
    --account julia.zhang:'Qwe@1245'

  bash scripts/deploy_forecast_services.sh \
    --sudo-password '881018' \
    --account mark:881018 \
    --proxy socks5h://127.0.0.1:1080
EOF
}

MINIFORGE_DIR="/opt/miniforge3"
SHARE_DIR="/srv/team-share"
STREAMLIT_APP=""
JUPYTERHUB_PORT="8000"
STREAMLIT_PORT="8501"
FILEBROWSER_PORT="8080"
PROXY_URL=""
SUDO_PASSWORD=""
ACCOUNTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account)
      [[ $# -ge 2 ]] || { echo "Missing value for --account" >&2; exit 2; }
      ACCOUNTS+=("$2")
      shift 2
      ;;
    --miniforge-dir)
      MINIFORGE_DIR="$2"
      shift 2
      ;;
    --share-dir)
      SHARE_DIR="$2"
      shift 2
      ;;
    --streamlit-app)
      STREAMLIT_APP="$2"
      shift 2
      ;;
    --jupyterhub-port)
      JUPYTERHUB_PORT="$2"
      shift 2
      ;;
    --streamlit-port)
      STREAMLIT_PORT="$2"
      shift 2
      ;;
    --filebrowser-port)
      FILEBROWSER_PORT="$2"
      shift 2
      ;;
    --proxy)
      PROXY_URL="$2"
      shift 2
      ;;
    --sudo-password)
      SUDO_PASSWORD="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#ACCOUNTS[@]} -eq 0 ]]; then
  echo "At least one --account USER:PASS is required." >&2
  exit 2
fi

if [[ -n "$PROXY_URL" ]]; then
  export http_proxy="$PROXY_URL"
  export https_proxy="$PROXY_URL"
  export HTTP_PROXY="$PROXY_URL"
  export HTTPS_PROXY="$PROXY_URL"
  export all_proxy="$PROXY_URL"
  export ALL_PROXY="$PROXY_URL"
fi

log() {
  printf '\n==> %s\n' "$*"
}

run_sudo() {
  if [[ ${EUID} -eq 0 ]]; then
    "$@"
  elif [[ -n "$SUDO_PASSWORD" ]]; then
    printf '%s\n' "$SUDO_PASSWORD" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

write_root_file() {
  local target="$1"
  local mode="${2:-0644}"
  local tmp
  tmp="$(mktemp)"
  cat > "$tmp"
  run_sudo install -m "$mode" "$tmp" "$target"
  rm -f "$tmp"
}

apt_proxy_opts=()
if [[ -n "$PROXY_URL" ]]; then
  apt_proxy_opts+=("-o" "Acquire::http::Proxy=$PROXY_URL")
  apt_proxy_opts+=("-o" "Acquire::https::Proxy=$PROXY_URL")
fi

apt_install() {
  run_sudo apt-get "${apt_proxy_opts[@]}" "$@"
}

ensure_supported_os() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "This script currently supports apt-based Linux distributions." >&2
    exit 1
  fi
}

create_accounts() {
  log "Create/update Linux users for JupyterHub"
  run_sudo groupadd -f users

  for account in "${ACCOUNTS[@]}"; do
    if [[ "$account" != *:* ]]; then
      echo "Invalid --account '$account'. Expected USER:PASS." >&2
      exit 2
    fi
    local user="${account%%:*}"
    local pass="${account#*:}"

    if ! id "$user" >/dev/null 2>&1; then
      run_sudo useradd -m -s /bin/bash "$user"
    fi
    run_sudo env NEW_USER="$user" NEW_PASS="$pass" \
      bash -c 'printf "%s:%s\n" "$NEW_USER" "$NEW_PASS" | chpasswd'
    run_sudo usermod -aG users "$user"
    run_sudo passwd -S "$user" >/dev/null
  done
}

install_os_packages() {
  log "Install OS packages"
  apt_install update
  apt_install install -y ca-certificates curl wget bzip2 tar gzip git less patch
}

install_miniforge() {
  log "Install Miniforge3"
  local arch mf_arch installer url
  arch="$(uname -m)"
  case "$arch" in
    x86_64) mf_arch="x86_64" ;;
    aarch64|arm64) mf_arch="aarch64" ;;
    *) echo "Unsupported architecture: $arch" >&2; exit 1 ;;
  esac

  installer="/tmp/Miniforge3-Linux-${mf_arch}.sh"
  url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${mf_arch}.sh"

  if [[ ! -x "$MINIFORGE_DIR/bin/conda" ]]; then
    curl -L --fail --retry 5 "$url" -o "$installer"
    run_sudo bash "$installer" -b -p "$MINIFORGE_DIR"
  fi

  run_sudo chmod -R a+rX "$MINIFORGE_DIR"

  run_sudo "$MINIFORGE_DIR/bin/conda" config --system --set auto_activate_base false || true
  run_sudo "$MINIFORGE_DIR/bin/conda" config --system --add channels conda-forge || true
  run_sudo "$MINIFORGE_DIR/bin/conda" config --system --set channel_priority strict
  if [[ -n "$PROXY_URL" ]]; then
    run_sudo "$MINIFORGE_DIR/bin/conda" config --system --set proxy_servers.http "$PROXY_URL"
    run_sudo "$MINIFORGE_DIR/bin/conda" config --system --set proxy_servers.https "$PROXY_URL"
  fi
}

install_python_services() {
  log "Install JupyterHub, JupyterLab, Streamlit and Node proxy"
  run_sudo "$MINIFORGE_DIR/bin/conda" install -n base -y -c conda-forge \
    jupyterhub jupyterlab notebook configurable-http-proxy nodejs streamlit pip
  run_sudo chmod -R a+rX "$MINIFORGE_DIR"
}

install_filebrowser() {
  log "Install File Browser"
  local arch fb_arch version url tmpdir
  arch="$(uname -m)"
  case "$arch" in
    x86_64) fb_arch="amd64" ;;
    aarch64|arm64) fb_arch="arm64" ;;
    *) echo "Unsupported architecture: $arch" >&2; exit 1 ;;
  esac

  version="$(curl -Ls -o /dev/null -w '%{url_effective}' \
    https://github.com/filebrowser/filebrowser/releases/latest | sed 's#.*/tag/##')"
  url="https://github.com/filebrowser/filebrowser/releases/download/${version}/linux-${fb_arch}-filebrowser.tar.gz"
  tmpdir="$(mktemp -d)"
  curl -L --fail --retry 5 "$url" -o "$tmpdir/filebrowser.tar.gz"
  tar -xzf "$tmpdir/filebrowser.tar.gz" -C "$tmpdir"
  run_sudo install -m 0755 "$tmpdir/filebrowser" /usr/local/bin/filebrowser
  rm -rf "$tmpdir"
}

configure_share_dir() {
  log "Configure public shared directory"
  run_sudo mkdir -p "$SHARE_DIR"
  run_sudo chgrp users "$SHARE_DIR"
  run_sudo chmod 2775 "$SHARE_DIR"
}

configure_filebrowser() {
  log "Configure File Browser"
  run_sudo mkdir -p /etc/filebrowser
  local db="/etc/filebrowser/filebrowser.db"

  if [[ ! -f "$db" ]]; then
    run_sudo filebrowser -d "$db" config init
  fi

  run_sudo filebrowser -d "$db" config set \
    --address 0.0.0.0 \
    --port "$FILEBROWSER_PORT" \
    --root "$SHARE_DIR" \
    --database "$db" \
    --minimumPasswordLength 6 \
    --fileMode 0o664 \
    --dirMode 0o775 >/dev/null

  for account in "${ACCOUNTS[@]}"; do
    local user="${account%%:*}"
    local pass="${account#*:}"
    if run_sudo filebrowser -d "$db" users ls | awk '{print $2}' | grep -qx "$user"; then
      run_sudo filebrowser -d "$db" users update "$user" --password "$pass" --perm.admin >/dev/null
    else
      run_sudo filebrowser -d "$db" users add "$user" "$pass" --perm.admin >/dev/null
    fi
  done
}

configure_jupyterhub() {
  log "Configure JupyterHub"
  run_sudo mkdir -p /etc/jupyterhub /var/lib/jupyterhub

  write_root_file /etc/jupyterhub/jupyterhub_config.py 0644 <<EOF
c.JupyterHub.bind_url = "http://0.0.0.0:${JUPYTERHUB_PORT}"
c.JupyterHub.cookie_secret_file = "/var/lib/jupyterhub/jupyterhub_cookie_secret"
c.JupyterHub.db_url = "sqlite:////var/lib/jupyterhub/jupyterhub.sqlite"
c.Spawner.default_url = "/lab"
c.Spawner.notebook_dir = "/home/{username}"
c.Authenticator.allow_all = True
EOF

  write_root_file /etc/systemd/system/jupyterhub.service 0644 <<EOF
[Unit]
Description=JupyterHub
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/var/lib/jupyterhub
Environment=PATH=${MINIFORGE_DIR}/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${MINIFORGE_DIR}/bin/jupyterhub -f /etc/jupyterhub/jupyterhub_config.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
}

configure_streamlit() {
  log "Configure Streamlit"
  local exec_start
  if [[ -n "$STREAMLIT_APP" ]]; then
    exec_start="${MINIFORGE_DIR}/bin/streamlit run ${STREAMLIT_APP} --server.address 0.0.0.0 --server.port ${STREAMLIT_PORT}"
  else
    exec_start="${MINIFORGE_DIR}/bin/streamlit hello --server.address 0.0.0.0 --server.port ${STREAMLIT_PORT}"
  fi

  write_root_file /etc/systemd/system/streamlit.service 0644 <<EOF
[Unit]
Description=Streamlit
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${SHARE_DIR}
Environment=PATH=${MINIFORGE_DIR}/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${exec_start}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
}

configure_filebrowser_service() {
  log "Configure File Browser service"
  write_root_file /etc/systemd/system/filebrowser.service 0644 <<EOF
[Unit]
Description=File Browser
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/filebrowser -d /etc/filebrowser/filebrowser.db
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
}

start_services() {
  log "Start services"
  run_sudo systemctl daemon-reload
  run_sudo systemctl enable --now jupyterhub.service streamlit.service filebrowser.service
  run_sudo systemctl restart jupyterhub.service streamlit.service filebrowser.service
}

verify() {
  log "Verify installation"
  git_version="$(git --version)"
  conda_version="$("$MINIFORGE_DIR/bin/conda" --version)"
  printf '%s\n' "$git_version"
  printf '%s\n' "$conda_version"
  "$MINIFORGE_DIR/bin/jupyterhub" --version
  "$MINIFORGE_DIR/bin/streamlit" version
  filebrowser version

  run_sudo systemctl is-active jupyterhub.service
  run_sudo systemctl is-active streamlit.service
  run_sudo systemctl is-active filebrowser.service

  curl -fsS "http://127.0.0.1:${JUPYTERHUB_PORT}/hub/login" >/dev/null
  curl -fsS "http://127.0.0.1:${STREAMLIT_PORT}" >/dev/null
  curl -fsS "http://127.0.0.1:${FILEBROWSER_PORT}" >/dev/null

  printf '\nInstallation summary:\n'
  printf '  Git:         installed (%s)\n' "$git_version"
  printf '  Miniforge:  installed (%s, %s)\n' "$conda_version" "$MINIFORGE_DIR"
  printf '  JupyterHub: http://<server-ip>:%s\n' "$JUPYTERHUB_PORT"
  printf '  FileBrowser (web upload enabled): http://<server-ip>:%s\n' "$FILEBROWSER_PORT"
  printf '  Streamlit:  http://<server-ip>:%s\n' "$STREAMLIT_PORT"
  printf '\nAccounts configured:\n'
  for account in "${ACCOUNTS[@]}"; do
    printf '  %s\n' "${account%%:*}"
  done
}

main() {
  ensure_supported_os
  install_os_packages
  create_accounts
  install_miniforge
  install_python_services
  install_filebrowser
  configure_share_dir
  configure_filebrowser
  configure_jupyterhub
  configure_streamlit
  configure_filebrowser_service
  start_services
  verify
}

main "$@"
