#!/usr/bin/env bash
# Install Kan portable miner as a systemd service.
#
# Run this from an unpacked portable package directory after editing kan.env:
#   cd ~/kan
#   ./install_service.sh
#
# Override defaults with env vars:
#   SERVICE_NAME=kan KAN_DIR=/opt/kan sudo -E ./install_service.sh
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-kan}"
KAN_DIR="${KAN_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ENV_FILE="${ENV_FILE:-${KAN_DIR}/kan.env}"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
LOGROTATE_FILE="/etc/logrotate.d/${SERVICE_NAME}"
USER_NAME="${KAN_USER:-$(id -un)}"
GROUP_NAME="${KAN_GROUP:-$(id -gn)}"

need_root() {
  if [ "$(id -u)" != "0" ]; then
    echo "ERROR: install_service.sh must run as root (use sudo -E)." >&2
    exit 2
  fi
}

write_default_env() {
  if [ -f "${ENV_FILE}" ]; then
    return 0
  fi
  cat > "${ENV_FILE}" <<'EOF'
# Kan miner service environment.
# Edit KAN_WALLET before starting the service.
KAN_POOL=stratum+tcp://prl.kryptex.network:7048
KAN_WALLET=
KAN_WORKER=rig01
KAN_BATCH=1000
KAN_EXTRA_ARGS=--cfg real --tc
KAN_RESTART=auto
KAN_RESTART_DELAY=15
# Optional GPU scoping:
# KAN_DEVICES=0,1
# CUDA_VISIBLE_DEVICES=0
EOF
  chmod 600 "${ENV_FILE}"
  chown "${USER_NAME}:${GROUP_NAME}" "${ENV_FILE}" 2>/dev/null || true
}

need_root
[ -x "${KAN_DIR}/run.sh" ] || { echo "ERROR: ${KAN_DIR}/run.sh not found or not executable" >&2; exit 2; }
[ -f "${KAN_DIR}/kan.service" ] || { echo "ERROR: ${KAN_DIR}/kan.service template missing" >&2; exit 2; }
[ -f "${KAN_DIR}/kan.logrotate" ] || { echo "ERROR: ${KAN_DIR}/kan.logrotate template missing" >&2; exit 2; }

write_default_env

sed \
  -e "s#__KAN_DIR__#${KAN_DIR}#g" \
  -e "s#__KAN_ENV__#${ENV_FILE}#g" \
  -e "s#__KAN_USER__#${USER_NAME}#g" \
  -e "s#__KAN_GROUP__#${GROUP_NAME}#g" \
  "${KAN_DIR}/kan.service" > "${SERVICE_FILE}"

sed \
  -e "s#__KAN_DIR__#${KAN_DIR}#g" \
  -e "s#__KAN_USER__#${USER_NAME}#g" \
  -e "s#__KAN_GROUP__#${GROUP_NAME}#g" \
  "${KAN_DIR}/kan.logrotate" > "${LOGROTATE_FILE}"

systemctl daemon-reload

echo "Service installed: ${SERVICE_FILE}"
echo "Environment:       ${ENV_FILE}"
echo "Logrotate:         ${LOGROTATE_FILE}"
echo
echo "Edit ${ENV_FILE} and set KAN_WALLET, then run:"
echo "  systemctl enable --now ${SERVICE_NAME}"
echo "  journalctl -u ${SERVICE_NAME} -f"
