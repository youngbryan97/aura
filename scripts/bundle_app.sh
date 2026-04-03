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
CONTENTS_DIR="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
EXECUTABLE_NAME="aura-launcher"
EXECUTABLE_PATH="${MACOS_DIR}/${EXECUTABLE_NAME}"
ICON_SOURCE="${ROOT_DIR}/aura_icon.icns"
ROOT_LINK="${RESOURCES_DIR}/aura-root"
ROOT_PATH_FALLBACK="${RESOURCES_DIR}/aura-root-path"
VERSION_FILE="${RESOURCES_DIR}/aura-version"
VERSION_FULL_FILE="${RESOURCES_DIR}/aura-version-full"
INFO_PLIST="${CONTENTS_DIR}/Info.plist"
LAUNCHER_SOURCE="${ROOT_DIR}/scripts/AuraLauncher.swift"

cd "${ROOT_DIR}"

echo "📦 Building ${APP_NAME} (live source mode)..."

if [ -x "${ROOT_DIR}/.venv/bin/python3" ] && [ -f "${ROOT_DIR}/scripts/build_launcher_icon.py" ]; then
    "${ROOT_DIR}/.venv/bin/python3" "${ROOT_DIR}/scripts/build_launcher_icon.py" >/dev/null
fi

if [ ! -f "${LAUNCHER_SOURCE}" ]; then
    echo "❌ Missing launcher source: ${LAUNCHER_SOURCE}"
    exit 1
fi

if ! command -v xcrun >/dev/null 2>&1; then
    echo "❌ xcrun is required to build the native Aura launcher."
    exit 1
fi

SWIFTC_PATH="$(xcrun --find swiftc 2>/dev/null || true)"
if [ -z "${SWIFTC_PATH}" ]; then
    echo "❌ swiftc is required to build the native Aura launcher."
    exit 1
fi

SDKROOT_PATH="$(xcrun --show-sdk-path --sdk macosx 2>/dev/null || true)"
rm -rf "${APP_DIR}"
mkdir -p "${MACOS_DIR}" "${RESOURCES_DIR}"

ln -sfn "${ROOT_DIR}" "${ROOT_LINK}"
printf '%s\n' "${ROOT_DIR}" > "${ROOT_PATH_FALLBACK}"

PYTHON_FOR_VERSION="${ROOT_DIR}/.venv/bin/python3"
if [ ! -x "${PYTHON_FOR_VERSION}" ]; then
    PYTHON_FOR_VERSION="$(command -v python3 || true)"
fi

APP_SEMVER="2026.3.31"
APP_FULL_VERSION="Aura Luna v${APP_SEMVER}"
if [ -n "${PYTHON_FOR_VERSION}" ]; then
    APP_SEMVER="$("${PYTHON_FOR_VERSION}" - <<'PY'
from core.version import VERSION
semver = VERSION.split("-", 1)[0]
print(semver)
PY
)"
    APP_FULL_VERSION="$("${PYTHON_FOR_VERSION}" - <<'PY'
from core.version import version_string
print(version_string("full"))
PY
)"
fi

printf '%s\n' "${APP_SEMVER}" > "${VERSION_FILE}"
printf '%s\n' "${APP_FULL_VERSION}" > "${VERSION_FULL_FILE}"

CLANG_MODULE_CACHE_PATH="${TMPDIR:-/tmp}/aura-launcher-clang-cache" xcrun swiftc \
    -O \
    -framework AppKit \
    -framework Foundation \
    "${LAUNCHER_SOURCE}" \
    -o "${EXECUTABLE_PATH}"

chmod +x "${EXECUTABLE_PATH}"

if [ -f "${ICON_SOURCE}" ]; then
    cp "${ICON_SOURCE}" "${RESOURCES_DIR}/Aura.icns"
fi

cat > "${INFO_PLIST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>Aura</string>
    <key>CFBundleExecutable</key>
    <string>aura-launcher</string>
    <key>CFBundleIconFile</key>
    <string>Aura.icns</string>
    <key>CFBundleIdentifier</key>
    <string>com.aura.desktop</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Aura</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${APP_SEMVER}</string>
    <key>CFBundleVersion</key>
    <string>${APP_SEMVER}</string>
    <key>NSCameraUsageDescription</key>
    <string>Aura can use the camera when you explicitly enable vision features.</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Aura can listen when you explicitly enable voice input.</string>
</dict>
</plist>
EOF

echo "✅ Built ${APP_DIR}"
echo "🧠 Live source link: ${ROOT_DIR}"
echo "✍️ Edit the repo normally — this launcher always runs the current workspace code."

if command -v codesign >/dev/null 2>&1; then
    codesign --force --sign - "${APP_DIR}" >/dev/null
fi

if [ -n "${INSTALL_PATH}" ]; then
    echo "📥 Installing ${APP_NAME} to ${INSTALL_PATH}..."
    rm -rf "${INSTALL_PATH}"
    cp -R "${APP_DIR}" "${INSTALL_PATH}"
    if command -v codesign >/dev/null 2>&1; then
        codesign --force --sign - "${INSTALL_PATH}" >/dev/null
    fi
    echo "✅ Installed ${INSTALL_PATH}"
fi
