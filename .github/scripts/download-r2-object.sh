#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: download-r2-object.sh OBJECT_PATH OUTPUT_PATH" >&2
  exit 2
fi

object_path="${1#/}"
output_path="$2"

: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID is required}"
: "${R2_BUCKET_ENDPOINT:?R2_BUCKET_ENDPOINT is required}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY is required}"

endpoint="${R2_BUCKET_ENDPOINT%/}"
case "${endpoint}" in
  https://*) ;;
  *)
    echo "download-r2-object: R2_BUCKET_ENDPOINT must be an https URL" >&2
    exit 1
    ;;
esac

endpoint_rest="${endpoint#https://}"
host="${endpoint_rest%%/*}"
base_path="${endpoint_rest#"${host}"}"
request_path="${base_path}/${object_path}"
url="https://${host}${request_path}"

region="${R2_REGION:-auto}"
service="s3"
amz_date="$(date -u +%Y%m%dT%H%M%SZ)"
date_stamp="${amz_date:0:8}"
payload_hash="$(printf '' | shasum -a 256 | awk '{print $1}')"
signed_headers="host;x-amz-content-sha256;x-amz-date"
canonical_headers="$(printf 'host:%s\nx-amz-content-sha256:%s\nx-amz-date:%s\n' "${host}" "${payload_hash}" "${amz_date}")"
canonical_request="$(printf 'GET\n%s\n\n%s\n%s\n%s' "${request_path}" "${canonical_headers}" "${signed_headers}" "${payload_hash}")"
credential_scope="${date_stamp}/${region}/${service}/aws4_request"
canonical_request_hash="$(printf '%s' "${canonical_request}" | shasum -a 256 | awk '{print $1}')"
string_to_sign="$(printf 'AWS4-HMAC-SHA256\n%s\n%s\n%s' "${amz_date}" "${credential_scope}" "${canonical_request_hash}")"

hmac_hex() {
  local key_opt="$1"
  local message="$2"
  printf '%s' "${message}" \
    | openssl dgst -sha256 -mac HMAC -macopt "${key_opt}" -binary \
    | xxd -p -c 256
}

k_date="$(hmac_hex "key:AWS4${R2_SECRET_ACCESS_KEY}" "${date_stamp}")"
k_region="$(hmac_hex "hexkey:${k_date}" "${region}")"
k_service="$(hmac_hex "hexkey:${k_region}" "${service}")"
k_signing="$(hmac_hex "hexkey:${k_service}" "aws4_request")"
signature="$(printf '%s' "${string_to_sign}" \
  | openssl dgst -sha256 -mac HMAC -macopt "hexkey:${k_signing}" \
  | awk '{print $2}')"

authorization="AWS4-HMAC-SHA256 Credential=${R2_ACCESS_KEY_ID}/${credential_scope}, SignedHeaders=${signed_headers}, Signature=${signature}"
tmp_path="${output_path}.tmp"

mkdir -p "$(dirname "${output_path}")"
curl \
  --fail \
  --silent \
  --show-error \
  --location \
  --retry 5 \
  --retry-all-errors \
  --retry-delay 2 \
  -H "Authorization: ${authorization}" \
  -H "x-amz-content-sha256: ${payload_hash}" \
  -H "x-amz-date: ${amz_date}" \
  --output "${tmp_path}" \
  "${url}"

chmod 600 "${tmp_path}"
mv "${tmp_path}" "${output_path}"
echo "download-r2-object: wrote ${output_path}"
