#!/usr/bin/env bash
# OPTIONAL one-time setup: unattended daily backup (systemd user timer).
# Workstation-friendly: fires 30 min after login (never during boot), then
# every 24h while the machine stays up; disk/CPU access is idle-priority so
# it yields to interactive work and training runs. Retries built in,
# per-machine subdir <hostname>-<machine-id8>.
# Optionally also syncs to Azure Blob — that requires persisting a
# container-scoped SAS to disk (0600, this machine only), which this script
# asks explicit consent for; decline and cloud backups stay manual.
#
#   ./setup_backup_service.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
CONF_DIR="$HOME/.config/falcon-vision-backup"
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$CONF_DIR" "$UNIT_DIR"

read -rp "NAS destination dir (e.g. /mnt/nas/falcon-backup): " NAS_DEST
[ -d "$(dirname "$NAS_DEST")" ] || echo "note: $(dirname "$NAS_DEST") not present right now — service will retry"
printf 'NAS_DEST=%q\n' "$NAS_DEST" > "$CONF_DIR/config"

read -rp "Also sync to Azure Blob? Requires storing a container SAS at $CONF_DIR/sas.url (0600). [y/N] " yn
if [ "${yn,,}" = "y" ]; then
  read -rsp "Container-scoped SAS URL: " SAS; echo
  (umask 077 && printf '%s' "$SAS" > "$CONF_DIR/sas.url")
  command -v rclone >/dev/null || echo "reminder: sudo apt install rclone"
else
  rm -f "$CONF_DIR/sas.url"
fi

cat > "$UNIT_DIR/falcon-backup.service" <<EOF
[Unit]
Description=Falcon Vision OD valuables backup (NAS + optional Azure)
After=network-online.target

[Service]
Type=oneshot
ExecStart=$REPO/backup_valuables.sh --from-config
Nice=19
CPUSchedulingPolicy=idle
IOSchedulingClass=idle
EOF

cat > "$UNIT_DIR/falcon-backup.timer" <<'EOF'
[Unit]
Description=Daily Falcon Vision OD backup

[Timer]
# Workstation pattern: fire well after login (not at boot), then daily while up.
OnStartupSec=30min
OnUnitActiveSec=24h

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now falcon-backup.timer
loginctl enable-linger "$USER" 2>/dev/null || true   # run even when logged out

echo
echo "installed. next run: $(systemctl --user list-timers falcon-backup.timer --no-pager | sed -n 2p)"
read -rp "Run a first backup now? [Y/n] " go
[ "${go,,}" != "n" ] && systemctl --user start falcon-backup.service && \
  journalctl --user -u falcon-backup.service --no-pager -n 5
echo "status any time:  systemctl --user status falcon-backup.timer"
echo "logs:             journalctl --user -u falcon-backup.service"
