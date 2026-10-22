#!/bin/bash
# Create a stable local code-signing identity for Aura.app rebuilds.
#
# macOS TCC permissions are tied to the requesting app identity. Ad-hoc signing
# changes on rebuild, so Accessibility/Screen Recording grants can appear active
# in System Settings while the current rebuilt app is still denied. This helper
# creates/imports a local self-signed code-signing certificate named
# "Aura Local Code Signing"; scripts/bundle_app.sh automatically uses it when
# present.

set -euo pipefail

IDENTITY_NAME="${AURA_LOCAL_CODESIGN_NAME:-Aura Local Code Signing}"
KEYCHAIN="${AURA_LOCAL_CODESIGN_KEYCHAIN:-${HOME}/Library/Keychains/login.keychain-db}"
P12_PASSWORD="${AURA_LOCAL_CODESIGN_P12_PASSWORD:-aura-local-code-signing}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aura-codesign.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

trust_identity_for_codesign() {
    if security find-certificate -c "${IDENTITY_NAME}" -p "${KEYCHAIN}" > "${WORK_DIR}/codesign.crt" 2>/dev/null; then
        security add-trusted-cert \
            -r trustRoot \
            -p codeSign \
            -k "${KEYCHAIN}" \
            "${WORK_DIR}/codesign.crt" \
            >/dev/null 2>&1 || true
    fi
    security set-key-partition-list \
        -l "${IDENTITY_NAME}" \
        -S apple-tool:,apple:,codesign: \
        -s \
        -k "${AURA_LOCAL_CODESIGN_KEYCHAIN_PASSWORD:-}" \
        "${KEYCHAIN}" \
        >/dev/null 2>&1 || true
}

verify_identity_can_sign() {
    local probe="${WORK_DIR}/codesign-probe"
    printf '#!/bin/sh\nexit 0\n' > "${probe}"
    chmod +x "${probe}"
    codesign --force --sign "${IDENTITY_NAME}" "${probe}" >/dev/null 2>&1
}

if security find-identity -v -p codesigning "${KEYCHAIN}" 2>/dev/null | grep -Fq "\"${IDENTITY_NAME}\""; then
    trust_identity_for_codesign
    if verify_identity_can_sign; then
        echo "✅ Existing code-signing identity found and verified: ${IDENTITY_NAME}"
        exit 0
    fi
    echo "❌ Existing identity cannot sign from this shell: ${IDENTITY_NAME}" >&2
    echo "   Remove it from Keychain Access and rerun this helper, or unlock/trust the key manually." >&2
    exit 1
fi

cat > "${WORK_DIR}/codesign.cnf" <<EOF
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_codesign
prompt = no

[req_distinguished_name]
CN = ${IDENTITY_NAME}

[v3_codesign]
basicConstraints = critical,CA:false
keyUsage = critical,digitalSignature
extendedKeyUsage = codeSigning
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid
EOF

openssl req \
    -newkey rsa:2048 \
    -nodes \
    -keyout "${WORK_DIR}/codesign.key" \
    -x509 \
    -days "${AURA_LOCAL_CODESIGN_DAYS:-3650}" \
    -out "${WORK_DIR}/codesign.crt" \
    -config "${WORK_DIR}/codesign.cnf" \
    >/dev/null 2>&1

openssl pkcs12 \
    -export \
    -inkey "${WORK_DIR}/codesign.key" \
    -in "${WORK_DIR}/codesign.crt" \
    -name "${IDENTITY_NAME}" \
    -certpbe PBE-SHA1-3DES \
    -keypbe PBE-SHA1-3DES \
    -macalg sha1 \
    -out "${WORK_DIR}/codesign.p12" \
    -passout "pass:${P12_PASSWORD}" \
    >/dev/null 2>&1

security import "${WORK_DIR}/codesign.p12" \
    -f pkcs12 \
    -k "${KEYCHAIN}" \
    -P "${P12_PASSWORD}" \
    -T /usr/bin/codesign \
    >/dev/null

trust_identity_for_codesign

if security find-identity -v -p codesigning "${KEYCHAIN}" 2>/dev/null | grep -Fq "\"${IDENTITY_NAME}\""; then
    if verify_identity_can_sign; then
        echo "✅ Created and verified code-signing identity: ${IDENTITY_NAME}"
    else
        echo "❌ Created identity, but codesign cannot use it from this shell." >&2
        exit 1
    fi
else
    echo "❌ Identity import finished, but codesign cannot see ${IDENTITY_NAME}" >&2
    exit 1
fi
