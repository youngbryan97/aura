#!/bin/bash
# scripts/bundle_app.sh
# Build a thin macOS app bundle that launches Aura from the live workspace.

set -euo pipefail

ROOT_DIR="$(cd -P "$(dirname "$0")/.." && pwd -P)"
DIST_DIR="${ROOT_DIR}/dist"
APP_BASENAME="${AURA_APP_NAME:-Aura}"
APP_NAME="${APP_BASENAME}.app"
APP_DIR="${DIST_DIR}/${APP_NAME}"
INSTALL_PATH="${AURA_INSTALL_PATH:-}"
DEFAULT_CODESIGN_IDENTITY="-"
if [ -n "${INSTALL_PATH}" ]; then
    DEFAULT_REQUIRE_STABLE_CODESIGN=1
else
    DEFAULT_REQUIRE_STABLE_CODESIGN=0
fi
AURA_REQUIRE_STABLE_CODESIGN="${AURA_REQUIRE_STABLE_CODESIGN:-${DEFAULT_REQUIRE_STABLE_CODESIGN}}"

run_with_timeout() {
    local timeout_s="$1"
    shift
    local status_file
    status_file="$(mktemp "${TMPDIR:-/tmp}/aura-run-timeout-status.XXXXXX")"
    (
        set +e
        "$@"
        printf '%s\n' "$?" > "${status_file}"
    ) &
    local child_pid="$!"
    local deadline=$((SECONDS + timeout_s))
    while [ ! -s "${status_file}" ]; do
        if ! kill -0 "${child_pid}" 2>/dev/null; then
            break
        fi
        if [ "${SECONDS}" -ge "${deadline}" ]; then
            kill -TERM "${child_pid}" 2>/dev/null || true
            sleep 0.2
            kill -KILL "${child_pid}" 2>/dev/null || true
            wait "${child_pid}" 2>/dev/null || true
            rm -f "${status_file}"
            return 124
        fi
        sleep 0.2
    done
    wait "${child_pid}" 2>/dev/null || true
    local status="1"
    if [ -s "${status_file}" ]; then
        status="$(cat "${status_file}")"
    fi
    rm -f "${status_file}"
    return "${status}"
}

# Prefer Aura's persistent local code-signing identity when it exists.  macOS TCC
# grants attach to the app identity, and ad-hoc signatures can drift on rebuilds.
AURA_AUTO_USE_LOCAL_CODESIGN="${AURA_AUTO_USE_LOCAL_CODESIGN:-1}"
if [ "${AURA_AUTO_USE_LOCAL_CODESIGN}" = "1" ] && command -v security >/dev/null 2>&1; then
    LOCAL_AURA_IDENTITY="$(
        security find-identity -v -p codesigning 2>/dev/null \
            | sed -n 's/.*"\(Aura Local Code Signing[^"]*\)".*/\1/p' \
            | head -n 1
    )"
    if [ -n "${LOCAL_AURA_IDENTITY}" ]; then
        SIGN_PROBE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aura-codesign-probe.XXXXXX")"
        trap 'rm -rf "${SIGN_PROBE_DIR}"' EXIT
        printf '#!/bin/sh\nexit 0\n' > "${SIGN_PROBE_DIR}/probe"
        chmod +x "${SIGN_PROBE_DIR}/probe"
        if run_with_timeout "${AURA_CODESIGN_PROBE_TIMEOUT_S:-8}" codesign --force --sign "${LOCAL_AURA_IDENTITY}" "${SIGN_PROBE_DIR}/probe" >/dev/null 2>&1; then
            DEFAULT_CODESIGN_IDENTITY="${LOCAL_AURA_IDENTITY}"
        elif [ "${AURA_REQUIRE_STABLE_CODESIGN}" = "1" ]; then
            echo "Local Aura code-signing identity exists but cannot sign from this shell." >&2
            exit 1
        else
            echo "Local Aura code-signing identity exists but cannot sign from this shell within ${AURA_CODESIGN_PROBE_TIMEOUT_S:-8}s; using ad-hoc signing." >&2
        fi
    fi
fi
CODESIGN_IDENTITY="${AURA_CODESIGN_IDENTITY:-${DEFAULT_CODESIGN_IDENTITY}}"
CONTENTS_DIR="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
EXECUTABLE_NAME="aura-launcher"
EXECUTABLE_PATH="${MACOS_DIR}/${EXECUTABLE_NAME}"
ICON_SOURCE="${ROOT_DIR}/aura_icon.icns"
ROOT_PATH_FALLBACK="${RESOURCES_DIR}/aura-root-path"
PYTHON_RUNTIME_FILE="${RESOURCES_DIR}/aura-python-path"
ENV_PATH_FILE="${RESOURCES_DIR}/aura-env-path"
MODELS_PATH_FILE="${RESOURCES_DIR}/aura-models-path"
FUSED_MODEL_PATH_FILE="${RESOURCES_DIR}/aura-fused-model-path"
VERSION_FILE="${RESOURCES_DIR}/aura-version"
VERSION_FULL_FILE="${RESOURCES_DIR}/aura-version-full"
PROVENANCE_FILE="${RESOURCES_DIR}/aura-launch-provenance.json"
INFO_PLIST="${CONTENTS_DIR}/Info.plist"
ENTITLEMENTS_PLIST="${DIST_DIR}/aura.entitlements"
LAUNCHER_SOURCE="${ROOT_DIR}/scripts/AuraLauncher.swift"
SCREEN_CAPTURE_POLICY_SOURCE="${ROOT_DIR}/config/screen_capture_privacy_policy.json"
SCREEN_CAPTURE_POLICY_RESOURCE="${RESOURCES_DIR}/screen_capture_privacy_policy.json"

cd "${ROOT_DIR}"

echo "Building ${APP_NAME} (live source mode)..."

PRIMARY_ROOT=""
GIT_COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
if [ -n "${GIT_COMMON_DIR}" ] && [ "$(basename "${GIT_COMMON_DIR}")" = ".git" ]; then
    PRIMARY_ROOT="$(dirname "${GIT_COMMON_DIR}")"
fi

PYTHON_RUNTIME=""
PYTHON_CANDIDATES=(
    "${AURA_PYTHON_RUNTIME:-}"
    "${ROOT_DIR}/.venv/bin/python3"
    "${ROOT_DIR}/.venv/bin/python"
)
if [ -n "${PRIMARY_ROOT}" ]; then
    PYTHON_CANDIDATES+=(
        "${PRIMARY_ROOT}/.venv/bin/python3"
        "${PRIMARY_ROOT}/.venv/bin/python"
    )
fi
for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if [ -z "${candidate}" ] || [ ! -x "${candidate}" ]; then
        continue
    fi
    if "${candidate}" -c 'import sys, httpx; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1; then
        PYTHON_RUNTIME="$("${candidate}" -c 'import sys; print(sys.executable)')"
        break
    fi
done
if [ -z "${PYTHON_RUNTIME}" ]; then
    echo "Aura needs the repository Python 3.12 environment with runtime dependencies installed." >&2
    echo "   Expected .venv under ${ROOT_DIR}${PRIMARY_ROOT:+ or ${PRIMARY_ROOT}}." >&2
    exit 1
fi

if [ -f "${ROOT_DIR}/scripts/build_launcher_icon.py" ]; then
    "${PYTHON_RUNTIME}" "${ROOT_DIR}/scripts/build_launcher_icon.py" >/dev/null
fi

if [ ! -f "${LAUNCHER_SOURCE}" ]; then
    echo "Missing launcher source: ${LAUNCHER_SOURCE}"
    exit 1
fi

if [ ! -f "${SCREEN_CAPTURE_POLICY_SOURCE}" ]; then
    echo "Missing screen-capture privacy policy: ${SCREEN_CAPTURE_POLICY_SOURCE}"
    exit 1
fi

if ! command -v xcrun >/dev/null 2>&1; then
    echo "xcrun is required to build the native Aura launcher."
    exit 1
fi

SWIFTC_PATH="$(xcrun --find swiftc 2>/dev/null || true)"
if [ -z "${SWIFTC_PATH}" ]; then
    echo "swiftc is required to build the native Aura launcher."
    exit 1
fi

SDKROOT_PATH="$(xcrun --show-sdk-path --sdk macosx 2>/dev/null || true)"
rm -rf "${APP_DIR}"
mkdir -p "${MACOS_DIR}" "${RESOURCES_DIR}"

printf '%s\n' "${ROOT_DIR}" > "${ROOT_PATH_FALLBACK}"
printf '%s\n' "${PYTHON_RUNTIME}" > "${PYTHON_RUNTIME_FILE}"
MODELS_ROOT="${ROOT_DIR}/models"
if [ -n "${PRIMARY_ROOT}" ] && [ -d "${PRIMARY_ROOT}/models" ]; then
    MODELS_ROOT="${PRIMARY_ROOT}/models"
fi
printf '%s\n' "${MODELS_ROOT}" > "${MODELS_PATH_FILE}"
FUSED_MODEL_ROOT="${ROOT_DIR}/training/fused-model"
if [ -n "${PRIMARY_ROOT}" ] && [ -d "${PRIMARY_ROOT}/training/fused-model" ]; then
    FUSED_MODEL_ROOT="${PRIMARY_ROOT}/training/fused-model"
fi
printf '%s\n' "${FUSED_MODEL_ROOT}" > "${FUSED_MODEL_PATH_FILE}"
ENV_SOURCE=""
for candidate in "${ROOT_DIR}/.env" "${PRIMARY_ROOT:+${PRIMARY_ROOT}/.env}"; do
    if [ -n "${candidate}" ] && [ -f "${candidate}" ]; then
        ENV_SOURCE="${candidate}"
        break
    fi
done
printf '%s\n' "${ENV_SOURCE}" > "${ENV_PATH_FILE}"
cp "${SCREEN_CAPTURE_POLICY_SOURCE}" "${SCREEN_CAPTURE_POLICY_RESOURCE}"

PYTHON_FOR_VERSION="${PYTHON_RUNTIME}"

APP_SEMVER="2026.3.31"
APP_FULL_VERSION="Aura Luna v${APP_SEMVER}"
APP_SEMVER="$("${PYTHON_FOR_VERSION}" - <<'PY'
from core.runtime.version import VERSION
semver = VERSION.split("-", 1)[0]
print(semver)
PY
)"
APP_FULL_VERSION="$("${PYTHON_FOR_VERSION}" - <<'PY'
from core.runtime.version import version_string
print(version_string("full"))
PY
)"

printf '%s\n' "${APP_SEMVER}" > "${VERSION_FILE}"
printf '%s\n' "${APP_FULL_VERSION}" > "${VERSION_FULL_FILE}"
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${PYTHON_FOR_VERSION}" -m core.runtime.launch_provenance emit \
    --root "${ROOT_DIR}" \
    --output "${PROVENANCE_FILE}" \
    --version "${APP_FULL_VERSION}" \
    --launcher-source "${LAUNCHER_SOURCE}" >/dev/null

CLANG_MODULE_CACHE_PATH="${TMPDIR:-/tmp}/aura-launcher-clang-cache" xcrun swiftc \
    -O \
    -framework AppKit \
    -framework CoreGraphics \
    -framework CryptoKit \
    -framework Foundation \
    -framework ScreenCaptureKit \
    -framework WebKit \
    "${LAUNCHER_SOURCE}" \
    -o "${EXECUTABLE_PATH}"

chmod +x "${EXECUTABLE_PATH}"

if [ -f "${ICON_SOURCE}" ]; then
    cp "${ICON_SOURCE}" "${RESOURCES_DIR}/Aura.icns"
fi

PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${PYTHON_FOR_VERSION}" - "${INFO_PLIST}" "${ENTITLEMENTS_PLIST}" "${APP_SEMVER}" <<'PY'
import plistlib
import sys
from pathlib import Path

from core.security.macos_bundle_manifest import (
    info_plist_overrides,
    write_entitlements_plist,
)

info_path = Path(sys.argv[1])
entitlements_path = Path(sys.argv[2])
app_semver = sys.argv[3]

payload = {
    "CFBundleDevelopmentRegion": "en",
    "CFBundleDisplayName": "Aura",
    "CFBundleExecutable": "aura-launcher",
    "CFBundleIconFile": "Aura.icns",
    "CFBundleIdentifier": "com.aura.desktop",
    "CFBundleInfoDictionaryVersion": "6.0",
    "CFBundleName": "Aura",
    "CFBundlePackageType": "APPL",
    "CFBundleShortVersionString": app_semver,
    "CFBundleVersion": app_semver,
    "NSHighResolutionCapable": True,
}
payload.update(info_plist_overrides())
with info_path.open("wb") as handle:
    plistlib.dump(payload, handle, sort_keys=True)
write_entitlements_plist(entitlements_path)
PY

echo "Built ${APP_DIR}"
echo "Live source link: ${ROOT_DIR}"
echo "Edit the repo normally — this launcher always runs the current workspace code."

sign_bundle() {
    local target="$1"
    local timeout_s="${AURA_CODESIGN_TIMEOUT_S:-45}"
    if run_with_timeout "${timeout_s}" codesign "${CODESIGN_ARGS[@]}" "${target}" >/dev/null; then
        return 0
    fi
    if [ "${CODESIGN_IDENTITY}" != "-" ] && [ "${AURA_REQUIRE_STABLE_CODESIGN}" != "1" ]; then
        echo "Codesigning ${target} with ${CODESIGN_IDENTITY} failed or timed out; falling back to ad-hoc signing." >&2
        local fallback_args=(--force --sign "-" --entitlements "${ENTITLEMENTS_PLIST}")
        run_with_timeout "${timeout_s}" codesign "${fallback_args[@]}" "${target}" >/dev/null
        return $?
    fi
    return 1
}

verify_signed_bundle() {
    local target="$1"
    local timeout_s="${AURA_CODESIGN_VERIFY_TIMEOUT_S:-20}"
    if ! run_with_timeout "${timeout_s}" codesign --verify --deep --strict --verbose=2 "${target}" >/dev/null 2>&1; then
        echo "Strict code-signature verification failed for ${target}." >&2
        return 1
    fi
}

if command -v codesign >/dev/null 2>&1; then
    if [ "${AURA_REQUIRE_STABLE_CODESIGN}" = "1" ] && [ "${CODESIGN_IDENTITY}" = "-" ]; then
        echo "A stable code-signing identity is required for installed Aura.app builds." >&2
        exit 1
    fi
    CODESIGN_ARGS=(--force --sign "${CODESIGN_IDENTITY}" --entitlements "${ENTITLEMENTS_PLIST}")
    if [ "${CODESIGN_IDENTITY}" != "-" ]; then
        CODESIGN_ARGS+=(--options runtime)
        if [ "${AURA_CODESIGN_TIMESTAMP:-0}" = "1" ]; then
            CODESIGN_ARGS+=(--timestamp)
        fi
    fi
    sign_bundle "${APP_DIR}"
    verify_signed_bundle "${APP_DIR}"
fi

if [ -n "${INSTALL_PATH}" ]; then
    echo "Installing ${APP_NAME} to ${INSTALL_PATH}..."
    rm -rf "${INSTALL_PATH}"
    cp -R "${APP_DIR}" "${INSTALL_PATH}"
    if command -v codesign >/dev/null 2>&1; then
        sign_bundle "${INSTALL_PATH}"
        verify_signed_bundle "${INSTALL_PATH}"
    fi
    echo "Installed ${INSTALL_PATH}"
fi
