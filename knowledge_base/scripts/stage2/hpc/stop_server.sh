#!/bin/bash
# stops run_server.sh and all its shards
cd "$(dirname "$0")"
P=$(cat logs/run_server.pid 2>/dev/null) && kill -TERM "$P" 2>/dev/null && echo "stopped $P and its shards" || echo "not running"
pkill -TERM -f '^python[0-9.]* s2_validate_agents\.py .*validation_agents_v3' 2>/dev/null; true
