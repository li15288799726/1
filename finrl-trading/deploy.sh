#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ──────────────────────────────────────────────
STRATEGY=""
CONFIG=""
DATA_DIR="data"
CRYPTO_DATA_DIR="data/crypto"
STOCK_DATA_DIR="data/stock"
START_DATE="2022-04-30"
END_DATE="2022-06-27"
MODE=""                  # backtest | train | paper | single
SINGLE_DATE=""
SKIP_DOWNLOAD=false
FREQ="W-FRI"
NO_FAST_TRACK=false
DRY_RUN=false
ACCOUNT=""               # Alpaca account name (paper mode)
ASSET_TYPE="both"        # crypto | stock | both
CRYPTO_TICKERS="BTCUSDT,ETHUSDT"
STOCK_TICKERS="AAPL,MSFT,GOOGL"
DRL_ALGORITHM="PPO"
EPISODES=100

# ── Strategy registry (add new strategies here) ──────────
# Format: strategy_name|config_path|runner_path|asset_type
STRATEGIES="
adaptive_rotation|src/strategies/AdaptiveRotationConf_v1.2.1.yaml|src/strategies/run_adaptive_rotation_strategy.py|stock
crypto_ppo|src/strategies/drl_strategy.py|src/strategies/run_drl_strategy.py|crypto
crypto_sac|src/strategies/drl_strategy.py|src/strategies/run_drl_strategy.py|crypto
crypto_ensemble|src/strategies/drl_strategy.py|src/strategies/run_drl_strategy.py|crypto
integrated_drl|src/strategies/drl_strategy.py|src/strategies/run_integrated_drl_strategy.py|both
"

# ── Helper Functions ──────────────────────────────────────
log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

log_warn() {
    echo "[WARN] $1" >&2
}

# ── Parse Command Line Arguments ──────────────────────────
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --strategy)
                STRATEGY="$2"
                shift 2
                ;;
            --asset-type)
                ASSET_TYPE="$2"
                shift 2
                ;;
            --crypto-tickers)
                CRYPTO_TICKERS="$2"
                shift 2
                ;;
            --stock-tickers)
                STOCK_TICKERS="$2"
                shift 2
                ;;
            --drl-algorithm)
                DRL_ALGORITHM="$2"
                shift 2
                ;;
            --episodes)
                EPISODES="$2"
                shift 2
                ;;
            --mode)
                MODE="$2"
                shift 2
                ;;
            --start-date)
                START_DATE="$2"
                shift 2
                ;;
            --end-date)
                END_DATE="$2"
                shift 2
                ;;
            --single-date)
                SINGLE_DATE="$2"
                shift 2
                ;;
            --skip-download)
                SKIP_DOWNLOAD=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --account)
                ACCOUNT="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# ── Show Help ─────────────────────────────────────────────
show_help() {
    cat << EOF
FinRL-Integrated Deployment Script

USAGE:
    ./deploy.sh [OPTIONS]

OPTIONS:
    --strategy NAME           Strategy to run (required)
    --asset-type TYPE        Asset type: crypto|stock|both (default: both)
    --crypto-tickers LIST    Comma-separated crypto tickers (default: BTCUSDT,ETHUSDT)
    --stock-tickers LIST     Comma-separated stock tickers (default: AAPL,MSFT,GOOGL)
    --drl-algorithm ALGO     DRL algorithm: PPO|SAC|DDPG|ENSEMBLE (default: PPO)
    --episodes N             Number of training episodes (default: 100)
    --mode MODE              Execution mode: backtest|train|paper|single
    --start-date DATE        Start date (default: 2022-04-30)
    --end-date DATE          End date (default: 2022-06-27)
    --single-date DATE        Single date for single mode
    --skip-download          Skip data download
    --dry-run                Show commands without executing
    --account NAME           Alpaca account name for paper trading
    --help                   Show this help message

AVAILABLE STRATEGIES:
    adaptive_rotation    Stock trading with adaptive rotation
    crypto_ppo          Crypto trading with PPO
    crypto_sac          Crypto trading with SAC
    crypto_ensemble     Crypto trading with ensemble
    integrated_drl       Integrated stock and crypto trading

EXAMPLES:
    # Train crypto PPO strategy
    ./deploy.sh --strategy crypto_ppo --mode train --asset-type crypto --episodes 200

    # Backtest integrated strategy
    ./deploy.sh --strategy integrated_drl --mode backtest --asset-type both

    # Paper trading crypto strategy
    ./deploy.sh --strategy crypto_sac --mode paper --asset-type crypto --account paper1

    # Single date prediction
    ./deploy.sh --strategy crypto_ppo --mode single --single-date 2022-06-27 --asset-type crypto
EOF
}

# ── Resolve Strategy ─────────────────────────────────────
resolve_strategy() {
    local input="$1"
    local name="" cfg="" runner="" asset_type=""
    
    # Check if input is a file path
    if [[ -f "$input" ]]; then
        cfg="$input"
        while IFS='|' read -r n c r a; do
            [[ -z "$n" ]] && continue
            if [[ "$c" == "$cfg" ]]; then
                name="$n"; runner="$r"; asset_type="$a"; break
            fi
        done <<< "$STRATEGIES"
        if [[ -z "$runner" ]]; then
            log_error "No runner registered for config '$cfg'"
            log_error "Register it in STRATEGIES in deploy.sh"
            return 1
        fi
    else
        while IFS='|' read -r n c r a; do
            [[ -z "$n" ]] && continue
            if [[ "$n" == "$input" ]]; then
                name="$n"; cfg="$c"; runner="$r"; asset_type="$a"; break
            fi
        done <<< "$STRATEGIES"
        if [[ -z "$runner" ]]; then
            log_error "Unknown strategy: '$input'"
            log_error "Available strategies:"
            while IFS='|' read -r n c r a; do
                [[ -z "$n" ]] && continue
                echo "  - $n ($a)"
            done <<< "$STRATEGIES"
            return 1
        fi
    fi
    
    STRATEGY_NAME="$name"
    STRATEGY_CONFIG="$cfg"
    STRATEGY_RUNNER="$runner"
    STRATEGY_ASSET_TYPE="$asset_type"
    
    log_info "Resolved strategy: $STRATEGY_NAME (asset type: $STRATEGY_ASSET_TYPE)"
}

# ── Validate Configuration ────────────────────────────────
validate_config() {
    # Check required parameters
    if [[ -z "$STRATEGY" ]]; then
        log_error "Strategy is required"
        show_help
        exit 1
    fi
    
    if [[ -z "$MODE" ]]; then
        log_error "Mode is required"
        show_help
        exit 1
    fi
    
    # Validate asset type compatibility
    if [[ "$ASSET_TYPE" != "crypto" && "$ASSET_TYPE" != "stock" && "$ASSET_TYPE" != "both" ]]; then
        log_error "Invalid asset type: $ASSET_TYPE"
        exit 1
    fi
    
    # Check strategy asset type compatibility
    if [[ "$STRATEGY_ASSET_TYPE" != "both" && "$ASSET_TYPE" != "$STRATEGY_ASSET_TYPE" ]]; then
        log_error "Strategy '$STRATEGY_NAME' is for $STRATEGY_ASSET_TYPE assets, but requested $ASSET_TYPE"
        exit 1
    fi
    
    # Validate mode compatibility
    case "$MODE" in
        backtest|train|paper|single)
            # Valid modes
            ;;
        *)
            log_error "Invalid mode: $MODE"
            exit 1
            ;;
    esac
    
    # Check dependencies for different modes
    if [[ "$MODE" == "paper" ]]; then
        if [[ -z "$ACCOUNT" ]]; then
            log_error "Account name is required for paper trading"
            exit 1
        fi
        
        # Check if Alpaca credentials are configured
        if [[ ! -f ".env" ]] || ! grep -q "APCA_API_KEY" ".env"; then
            log_error "Alpaca API credentials not configured in .env"
            exit 1
        fi
    fi
    
    # Check data availability
    if [[ "$MODE" == "backtest" || "$MODE" == "train" ]]; then
        if [[ "$SKIP_DOWNLOAD" == false ]]; then
            download_data
        fi
    fi
}

# ── Download Data ────────────────────────────────────────
download_data() {
    log_info "Downloading data for $ASSET_TYPE assets..."
    
    if [[ "$MODE" == "train" || "$MODE" == "backtest" ]]; then
        # Training data
        if [[ "$ASSET_TYPE" == "crypto" || "$ASSET_TYPE" == "both" ]]; then
            log_info "Downloading crypto training data..."
            python3 src/data/download_crypto_data.py --tickers "$CRYPTO_TICKERS" --start "$START_DATE" --end "$END_DATE"
        fi
        
        if [[ "$ASSET_TYPE" == "stock" || "$ASSET_TYPE" == "both" ]]; then
            log_info "Downloading stock training data..."
            python3 src/data/download_stock_data.py --tickers "$STOCK_TICKERS" --start "$START_DATE" --end "$END_DATE"
        fi
    fi
    
    if [[ "$MODE" == "paper" ]]; then
        # Paper trading data (recent data)
        log_info "Downloading recent data for paper trading..."
        if [[ "$ASSET_TYPE" == "crypto" || "$ASSET_TYPE" == "both" ]]; then
            python3 src/data/download_crypto_data.py --tickers "$CRYPTO_TICKERS" --start "2024-01-01" --end "2024-12-31"
        fi
    fi
}

# ── Execute Strategy ──────────────────────────────────────
execute_strategy() {
    local cmd="python3 $STRATEGY_RUNNER"
    
    # Build command based on mode and strategy
    case "$MODE" in
        backtest)
            cmd="$cmd --mode backtest --config $STRATEGY_CONFIG --start-date $START_DATE --end-date $END_DATE"
            ;;
        train)
            cmd="$cmd --mode train --config $STRATEGY_CONFIG --episodes $EPISODES --start-date $START_DATE --end-date $END_DATE"
            ;;
        paper)
            cmd="$cmd --mode paper --config $STRATEGY_CONFIG --account $ACCOUNT --start-date $START_DATE --end-date $END_DATE"
            ;;
        single)
            if [[ -z "$SINGLE_DATE" ]]; then
                log_error "Single date is required for single mode"
                exit 1
            fi
            cmd="$cmd --mode single --config $STRATEGY_CONFIG --date $SINGLE_DATE"
            ;;
    esac
    
    # Add asset-specific parameters
    if [[ "$ASSET_TYPE" == "crypto" ]]; then
        cmd="$cmd --crypto-tickers $CRYPTO_TICKERS"
    elif [[ "$ASSET_TYPE" == "stock" ]]; then
        cmd="$cmd --stock-tickers $STOCK_TICKERS"
    else
        cmd="$cmd --crypto-tickers $CRYPTO_TICKERS --stock-tickers $STOCK_TICKERS"
    fi
    
    # Add DRL algorithm for DRL strategies
    if [[ "$STRATEGY_NAME" == *"crypto"* || "$STRATEGY_NAME" == "integrated_drl" ]]; then
        cmd="$cmd --drl-algorithm $DRL_ALGORITHM"
    fi
    
    # Execute command
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Dry run: $cmd"
    else
        log_info "Executing: $cmd"
        eval "$cmd"
    fi
}

# ── Main Execution ───────────────────────────────────────
main() {
    parse_args "$@"
    
    if [[ -z "$STRATEGY" ]]; then
        show_help
        exit 1
    fi
    
    log_info "FinRL-Integrated Deployment Script"
    log_info "=================================="
    log_info "Strategy: $STRATEGY"
    log_info "Asset Type: $ASSET_TYPE"
    log_info "Mode: $MODE"
    log_info "Start Date: $START_DATE"
    log_info "End Date: $END_DATE"
    
    # Resolve strategy
    if ! resolve_strategy "$STRATEGY"; then
        exit 1
    fi
    
    # Validate configuration
    validate_config
    
    # Execute strategy
    execute_strategy
    
    log_info "Execution completed!"
}

# ── Entry Point ───────────────────────────────────────────
main "$@"