#!/usr/bin/env bash
# Integration test for MLShield Docker stack
# Usage: docker compose up --build -d && bash tests/integration/test_docker.sh

set -euo pipefail

API_URL="${MLSHIELD_URL:-http://localhost:8000}"
PASS=0
FAIL=0

check() {
    local name="$1"
    local result="$2"
    if [ "$result" = "0" ]; then
        echo "  PASS: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $name"
        FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo "  MLShield Docker Integration Tests"
echo "========================================"
echo "  Target: $API_URL"
echo ""

# Wait for service to be ready
echo "Waiting for MLShield to start..."
for i in $(seq 1 30); do
    if curl -sf "$API_URL/health" > /dev/null 2>&1; then
        echo "  MLShield is ready!"
        break
    fi
    if [ "$i" = "30" ]; then
        echo "  FAIL: MLShield did not start within 30 seconds"
        exit 1
    fi
    sleep 1
done

echo ""
echo "--- Health Check ---"
HEALTH=$(curl -sf "$API_URL/health")
echo "$HEALTH" | python -m json.tool 2>/dev/null || echo "$HEALTH"
check "Health endpoint returns 200" "$?"

STATUS=$(echo "$HEALTH" | python -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "")
[ "$STATUS" = "healthy" ] && check "Status is healthy" "0" || check "Status is healthy" "1"

echo ""
echo "--- Submit Benign Event ---"
RESULT=$(curl -sf -X POST "$API_URL/api/v1/events" \
    -H "Content-Type: application/json" \
    -d '{"action": "k8s_get", "resource": "pods/health", "job_id": "test-1"}')
check "Benign event accepted" "$?"

IS_THREAT=$(echo "$RESULT" | python -c "import sys,json; print(json.load(sys.stdin)['is_threat'])" 2>/dev/null || echo "")
[ "$IS_THREAT" = "False" ] && check "Benign event not flagged" "0" || check "Benign event not flagged" "1"

echo ""
echo "--- Submit Threat Event ---"
RESULT=$(curl -sf -X POST "$API_URL/api/v1/events" \
    -H "Content-Type: application/json" \
    -d '{"action": "k8s_get", "resource": "secrets/aws-credentials", "job_id": "test-2"}')
check "Threat event accepted" "$?"

IS_THREAT=$(echo "$RESULT" | python -c "import sys,json; print(json.load(sys.stdin)['is_threat'])" 2>/dev/null || echo "")
[ "$IS_THREAT" = "True" ] && check "Credential access detected as threat" "0" || check "Credential access detected as threat" "1"

echo ""
echo "--- Submit Batch ---"
RESULT=$(curl -sf -X POST "$API_URL/api/v1/events/batch" \
    -H "Content-Type: application/json" \
    -d '{"events": [
        {"action": "k8s_get", "resource": "pods/data-loader", "job_id": "batch-1"},
        {"action": "network_egress", "resource": "pods/training", "job_id": "batch-2", "details": {"destination": "attacker.s3.amazonaws.com"}}
    ]}')
check "Batch endpoint works" "$?"

THREATS=$(echo "$RESULT" | python -c "import sys,json; print(json.load(sys.stdin)['threats_found'])" 2>/dev/null || echo "0")
[ "$THREATS" -ge "1" ] && check "Batch detected at least 1 threat" "0" || check "Batch detected at least 1 threat" "1"

echo ""
echo "--- Stats Endpoint ---"
curl -sf "$API_URL/api/v1/stats" > /dev/null
check "Stats endpoint works" "$?"

echo ""
echo "--- Alerts Endpoint ---"
ALERTS=$(curl -sf "$API_URL/api/v1/alerts")
check "Alerts endpoint works" "$?"

TOTAL=$(echo "$ALERTS" | python -c "import sys,json; print(json.load(sys.stdin)['total'])" 2>/dev/null || echo "0")
[ "$TOTAL" -ge "1" ] && check "Alerts contain threat detections" "0" || check "Alerts contain threat detections" "1"

echo ""
echo "--- Metrics Endpoint ---"
METRICS=$(curl -sf "$API_URL/metrics")
check "Prometheus metrics endpoint works" "$?"
echo "$METRICS" | grep -q "mlshield_events_total" && check "Metrics contain event counters" "0" || check "Metrics contain event counters" "1"

echo ""
echo "--- Dashboard ---"
curl -sf "$API_URL/" | grep -q "MLShield" > /dev/null
check "Dashboard serves HTML" "$?"

echo ""
echo "========================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "========================================"

[ "$FAIL" = "0" ] && exit 0 || exit 1
