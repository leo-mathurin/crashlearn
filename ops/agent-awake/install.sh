#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo 'lancer avec sudo' >&2
  exit 1
fi

src=$(cd "$(dirname "$0")" && pwd)
install -D -m 755 "$src/agent-awake" /usr/local/bin/agent-awake
install -D -m 755 "$src/agent-awake-daemon" /usr/local/lib/tour/agent-awake-daemon
install -D -m 755 "$src/agent-awake-control" /usr/local/sbin/agent-awake-control
install -D -m 644 "$src/agent-awake.env" /etc/tour/agent-awake.env
install -D -m 644 "$src/agent-awake.service" /etc/systemd/system/agent-awake.service
printf '%s\n' 'leo ALL=(root) NOPASSWD: /usr/local/sbin/agent-awake-control on, /usr/local/sbin/agent-awake-control off' \
  > /etc/sudoers.d/92-agent-awake
chmod 440 /etc/sudoers.d/92-agent-awake
visudo -cf /etc/sudoers.d/92-agent-awake >/dev/null
systemctl daemon-reload

if [ "${1:-}" != --install-only ]; then
  systemctl enable --now agent-awake.service
fi
