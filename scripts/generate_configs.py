import yaml
from pathlib import Path
from quant_engine.config import load_config

# Define configurations
configs = {}

# 1. Nifty Intraday Momentum
configs["nifty_intraday_momentum"] = {
    "name": "Nifty Intraday Momentum",
    "description": "High-speed trend-following and momentum trading on NIFTY index (NSE)",
    "trading_styles": ["intraday"],
    "style_overrides": {
        "intraday": {
            "max_hold_bars": 75,
            "forced_exit_time": "15:15",
            "product_type": "MIS",
            "min_trades": 150
        }
    },
    "data": {
        "openalgo": {
            "host": "http://127.0.0.1:5000",
            "api_key": "${OPENALGO_API_KEY}",
            "source": "db"
        },
        "symbols": [
            {"symbol": "NIFTY", "exchange": "NSE"}
        ],
        "timeframes": ["5m", "15m"],
        "start_date": "2021-01-01",
        "end_date": "2026-06-01",
        "train_pct": 0.70,
        "validation_pct": 0.15,
        "test_pct": 0.15
    },
    "generation": {
        "mode": "guided",
        "target_count": 30000,
        "max_conditions_per_entry": 3,
        "allow_short": True,
        "indicator_categories": ["trend", "momentum"],
        "multi_timeframe": False
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 4,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 150,
            "min_sharpe": 1.2,
            "max_drawdown": 0.20,
            "min_profit_factor": 1.4,
            "min_win_rate": 0.40,
            "min_cagr": 0.15
        },
        "validation": {
            "max_oos_sharpe_decay": 0.35,
            "min_walk_forward_consistency": 0.65,
            "monte_carlo_confidence": 0.95,
            "param_stability_tolerance": 0.20
        }
    },
    "evolution": {
        "enabled": True,
        "generations": 6,
        "population_size": 300,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
        "elitism_pct": 0.1,
        "tournament_size": 4
    },
    "ranking": {
        "mode": "robustness_first",
        "objectives": [
            {"metric": "sharpe", "weight": 0.35, "direction": "maximize"},
            {"metric": "sortino", "weight": 0.20, "direction": "maximize"},
            {"metric": "max_drawdown_pct", "weight": 0.20, "direction": "minimize"},
            {"metric": "profit_factor", "weight": 0.15, "direction": "maximize"},
            {"metric": "cagr", "weight": 0.10, "direction": "maximize"}
        ],
        "export_top_n": 15,
        "export_per_category": 3
    },
    "cost_model": {
        "commission_pct": 0.03,
        "slippage_pct": 0.02,
        "min_commission": 20.0
    },
    "output": {
        "storage": "csv",
        "base_dir": "./data/runs",
        "export_dir": "./data/exports",
        "save_all_candidates": True
    }
}

# 2. BankNifty Intraday Mean Reversion
configs["banknifty_intraday_mean_reversion"] = {
    "name": "BankNifty Intraday Mean Reversion",
    "description": "Volatility and mean reversion setups on highly volatile BANKNIFTY index",
    "trading_styles": ["intraday"],
    "style_overrides": {
        "intraday": {
            "max_hold_bars": 75,
            "forced_exit_time": "15:15",
            "product_type": "MIS",
            "min_trades": 180
        }
    },
    "data": {
        "openalgo": {
            "host": "http://127.0.0.1:5000",
            "api_key": "${OPENALGO_API_KEY}",
            "source": "db"
        },
        "symbols": [
            {"symbol": "BANKNIFTY", "exchange": "NSE"}
        ],
        "timeframes": ["5m", "15m"],
        "start_date": "2021-01-01",
        "end_date": "2026-06-01",
        "train_pct": 0.70,
        "validation_pct": 0.15,
        "test_pct": 0.15
    },
    "generation": {
        "mode": "guided",
        "target_count": 30000,
        "max_conditions_per_entry": 3,
        "allow_short": True,
        "indicator_categories": ["volatility", "momentum"],
        "multi_timeframe": False
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 4,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 180,
            "min_sharpe": 1.1,
            "max_drawdown": 0.22,
            "min_profit_factor": 1.35,
            "min_win_rate": 0.42,
            "min_cagr": 0.15
        },
        "validation": {
            "max_oos_sharpe_decay": 0.40,
            "min_walk_forward_consistency": 0.60,
            "monte_carlo_confidence": 0.95,
            "param_stability_tolerance": 0.25
        }
    },
    "evolution": {
        "enabled": True,
        "generations": 5,
        "population_size": 300,
        "mutation_rate": 0.35,
        "crossover_rate": 0.5,
        "elitism_pct": 0.1,
        "tournament_size": 4
    },
    "ranking": {
        "mode": "robustness_first",
        "objectives": [
            {"metric": "sharpe", "weight": 0.30, "direction": "maximize"},
            {"metric": "sortino", "weight": 0.20, "direction": "maximize"},
            {"metric": "max_drawdown_pct", "weight": 0.25, "direction": "minimize"},
            {"metric": "profit_factor", "weight": 0.15, "direction": "maximize"},
            {"metric": "cagr", "weight": 0.10, "direction": "maximize"}
        ],
        "export_top_n": 15,
        "export_per_category": 3
    },
    "cost_model": {
        "commission_pct": 0.03,
        "slippage_pct": 0.02,
        "min_commission": 20.0
    },
    "output": {
        "storage": "csv",
        "base_dir": "./data/runs",
        "export_dir": "./data/exports",
        "save_all_candidates": True
    }
}

# 3. Indian Bluechips Swing Trend
configs["indian_bluechips_swing_trend"] = {
    "name": "Indian Bluechips Swing Trend",
    "description": "Medium-term swing trading across top NSE liquid blue-chips",
    "trading_styles": ["swing"],
    "style_overrides": {
        "swing": {
            "min_hold_bars": 10,
            "max_hold_bars": 360,
            "product_type": "CNC",
            "min_trades": 50
        }
    },
    "data": {
        "openalgo": {
            "host": "http://127.0.0.1:5000",
            "api_key": "${OPENALGO_API_KEY}",
            "source": "db"
        },
        "symbols": [
            {"symbol": "RELIANCE", "exchange": "NSE"},
            {"symbol": "TCS", "exchange": "NSE"},
            {"symbol": "INFOSYS", "exchange": "NSE"},
            {"symbol": "HDFCBANK", "exchange": "NSE"},
            {"symbol": "ICICIBANK", "exchange": "NSE"},
            {"symbol": "SBIN", "exchange": "NSE"},
            {"symbol": "LT", "exchange": "NSE"},
            {"symbol": "ITC", "exchange": "NSE"},
            {"symbol": "TATAMOTORS", "exchange": "NSE"},
            {"symbol": "BHARTIENTL", "exchange": "NSE"}
        ],
        "timeframes": ["1h", "1d"],
        "start_date": "2021-01-01",
        "end_date": "2026-06-01",
        "train_pct": 0.70,
        "validation_pct": 0.15,
        "test_pct": 0.15
    },
    "generation": {
        "mode": "random",
        "target_count": 50000,
        "max_conditions_per_entry": 3,
        "allow_short": False,
        "indicator_categories": ["trend", "volume"],
        "multi_timeframe": True
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 4,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 50,
            "min_sharpe": 1.2,
            "max_drawdown": 0.18,
            "min_profit_factor": 1.4,
            "min_win_rate": 0.45,
            "min_cagr": 0.18
        },
        "validation": {
            "max_oos_sharpe_decay": 0.30,
            "min_walk_forward_consistency": 0.70,
            "monte_carlo_confidence": 0.95,
            "param_stability_tolerance": 0.20
        }
    },
    "evolution": {
        "enabled": True,
        "generations": 5,
        "population_size": 400,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
        "elitism_pct": 0.1,
        "tournament_size": 5
    },
    "ranking": {
        "mode": "robustness_first",
        "objectives": [
            {"metric": "sharpe", "weight": 0.35, "direction": "maximize"},
            {"metric": "sortino", "weight": 0.25, "direction": "maximize"},
            {"metric": "max_drawdown_pct", "weight": 0.20, "direction": "minimize"},
            {"metric": "profit_factor", "weight": 0.10, "direction": "maximize"},
            {"metric": "cagr", "weight": 0.10, "direction": "maximize"}
        ],
        "export_top_n": 20,
        "export_per_category": 4
    },
    "cost_model": {
        "commission_pct": 0.12,
        "slippage_pct": 0.03,
        "min_commission": 0.0
    },
    "output": {
        "storage": "csv",
        "base_dir": "./data/runs",
        "export_dir": "./data/exports",
        "save_all_candidates": True
    }
}

# 4. Midcap High Alpha Swing
configs["midcap_high_alpha_swing"] = {
    "name": "Midcap High Alpha Swing",
    "description": "High-momentum swing breakout trading targeting alpha in midcap growth stocks",
    "trading_styles": ["swing"],
    "style_overrides": {
        "swing": {
            "min_hold_bars": 10,
            "max_hold_bars": 360,
            "product_type": "CNC",
            "min_trades": 30
        }
    },
    "data": {
        "openalgo": {
            "host": "http://127.0.0.1:5000",
            "api_key": "${OPENALGO_API_KEY}",
            "source": "db"
        },
        "symbols": [
            {"symbol": "TATAPOWER", "exchange": "NSE"},
            {"symbol": "JINDALSTEL", "exchange": "NSE"},
            {"symbol": "ASHOKLEY", "exchange": "NSE"},
            {"symbol": "BEL", "exchange": "NSE"},
            {"symbol": "HAL", "exchange": "NSE"}
        ],
        "timeframes": ["1d"],
        "start_date": "2021-01-01",
        "end_date": "2026-06-01",
        "train_pct": 0.70,
        "validation_pct": 0.15,
        "test_pct": 0.15
    },
    "generation": {
        "mode": "random",
        "target_count": 40000,
        "max_conditions_per_entry": 3,
        "allow_short": False,
        "indicator_categories": ["trend", "momentum", "volume"],
        "multi_timeframe": False
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 5,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 30,
            "min_sharpe": 1.3,
            "max_drawdown": 0.25,
            "min_profit_factor": 1.5,
            "min_win_rate": 0.40,
            "min_cagr": 0.25
        },
        "validation": {
            "max_oos_sharpe_decay": 0.35,
            "min_walk_forward_consistency": 0.65,
            "monte_carlo_confidence": 0.95,
            "param_stability_tolerance": 0.25
        }
    },
    "evolution": {
        "enabled": True,
        "generations": 5,
        "population_size": 400,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
        "elitism_pct": 0.1,
        "tournament_size": 4
    },
    "ranking": {
        "mode": "robustness_first",
        "objectives": [
            {"metric": "sharpe", "weight": 0.30, "direction": "maximize"},
            {"metric": "sortino", "weight": 0.20, "direction": "maximize"},
            {"metric": "max_drawdown_pct", "weight": 0.15, "direction": "minimize"},
            {"metric": "profit_factor", "weight": 0.15, "direction": "maximize"},
            {"metric": "cagr", "weight": 0.20, "direction": "maximize"}
        ],
        "export_top_n": 20,
        "export_per_category": 5
    },
    "cost_model": {
        "commission_pct": 0.12,
        "slippage_pct": 0.04,
        "min_commission": 0.0
    },
    "output": {
        "storage": "csv",
        "base_dir": "./data/runs",
        "export_dir": "./data/exports",
        "save_all_candidates": True
    }
}

# 5. Heavyweight Hybrid Intraday Swing
configs["heavyweight_hybrid_intraday_swing"] = {
    "name": "Heavyweight Hybrid Intraday Swing",
    "description": "Multi-style intraday and swing strategy search on RELIANCE and HDFCBANK",
    "trading_styles": ["intraday", "swing"],
    "style_overrides": {
        "intraday": {
            "max_hold_bars": 75,
            "forced_exit_time": "15:15",
            "product_type": "MIS",
            "min_trades": 100
        },
        "swing": {
            "min_hold_bars": 10,
            "max_hold_bars": 360,
            "product_type": "CNC",
            "min_trades": 30
        }
    },
    "data": {
        "openalgo": {
            "host": "http://127.0.0.1:5000",
            "api_key": "${OPENALGO_API_KEY}",
            "source": "db"
        },
        "symbols": [
            {"symbol": "RELIANCE", "exchange": "NSE"},
            {"symbol": "HDFCBANK", "exchange": "NSE"}
        ],
        "timeframes": ["15m", "1h"],
        "start_date": "2021-01-01",
        "end_date": "2026-06-01",
        "train_pct": 0.70,
        "validation_pct": 0.15,
        "test_pct": 0.15
    },
    "generation": {
        "mode": "guided",
        "target_count": 40000,
        "max_conditions_per_entry": 4,
        "allow_short": True,
        "indicator_categories": ["trend", "momentum", "volatility", "volume"],
        "multi_timeframe": True
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 5,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 50,
            "min_sharpe": 1.1,
            "max_drawdown": 0.20,
            "min_profit_factor": 1.35,
            "min_win_rate": 0.40,
            "min_cagr": 0.15
        },
        "validation": {
            "max_oos_sharpe_decay": 0.35,
            "min_walk_forward_consistency": 0.65,
            "monte_carlo_confidence": 0.95,
            "param_stability_tolerance": 0.20
        }
    },
    "evolution": {
        "enabled": True,
        "generations": 5,
        "population_size": 300,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
        "elitism_pct": 0.1,
        "tournament_size": 4
    },
    "ranking": {
        "mode": "robustness_first",
        "objectives": [
            {"metric": "sharpe", "weight": 0.30, "direction": "maximize"},
            {"metric": "sortino", "weight": 0.20, "direction": "maximize"},
            {"metric": "max_drawdown_pct", "weight": 0.20, "direction": "minimize"},
            {"metric": "profit_factor", "weight": 0.15, "direction": "maximize"},
            {"metric": "cagr", "weight": 0.15, "direction": "maximize"}
        ],
        "export_top_n": 20,
        "export_per_category": 4
    },
    "cost_model": {
        "commission_pct": 0.06,
        "slippage_pct": 0.02,
        "min_commission": 10.0
    },
    "output": {
        "storage": "csv",
        "base_dir": "./data/runs",
        "export_dir": "./data/exports",
        "save_all_candidates": True
    }
}

# 6. Sector IT Momentum Intraday
configs["sector_it_momentum_intraday"] = {
    "name": "Sector IT Momentum Intraday",
    "description": "Intraday momentum strategy for high beta IT sector stocks",
    "trading_styles": ["intraday"],
    "style_overrides": {
        "intraday": {
            "max_hold_bars": 75,
            "forced_exit_time": "15:15",
            "product_type": "MIS",
            "min_trades": 150
        }
    },
    "data": {
        "openalgo": {
            "host": "http://127.0.0.1:5000",
            "api_key": "${OPENALGO_API_KEY}",
            "source": "db"
        },
        "symbols": [
            {"symbol": "TCS", "exchange": "NSE"},
            {"symbol": "INFOSYS", "exchange": "NSE"},
            {"symbol": "WIPRO", "exchange": "NSE"},
            {"symbol": "HCLTECH", "exchange": "NSE"},
            {"symbol": "TECHM", "exchange": "NSE"}
        ],
        "timeframes": ["15m"],
        "start_date": "2021-01-01",
        "end_date": "2026-06-01",
        "train_pct": 0.70,
        "validation_pct": 0.15,
        "test_pct": 0.15
    },
    "generation": {
        "mode": "guided",
        "target_count": 30000,
        "max_conditions_per_entry": 3,
        "allow_short": True,
        "indicator_categories": ["momentum", "trend"],
        "multi_timeframe": False
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 4,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 150,
            "min_sharpe": 1.0,
            "max_drawdown": 0.15,
            "min_profit_factor": 1.30,
            "min_win_rate": 0.38,
            "min_cagr": 0.12
        },
        "validation": {
            "max_oos_sharpe_decay": 0.35,
            "min_walk_forward_consistency": 0.60,
            "monte_carlo_confidence": 0.95,
            "param_stability_tolerance": 0.20
        }
    },
    "evolution": {
        "enabled": True,
        "generations": 5,
        "population_size": 250,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
        "elitism_pct": 0.1,
        "tournament_size": 4
    },
    "ranking": {
        "mode": "weighted",
        "objectives": [
            {"metric": "sharpe", "weight": 0.40, "direction": "maximize"},
            {"metric": "profit_factor", "weight": 0.30, "direction": "maximize"},
            {"metric": "max_drawdown_pct", "weight": 0.30, "direction": "minimize"}
        ],
        "export_top_n": 15,
        "export_per_category": 3
    },
    "cost_model": {
        "commission_pct": 0.03,
        "slippage_pct": 0.02,
        "min_commission": 20.0
    },
    "output": {
        "storage": "csv",
        "base_dir": "./data/runs",
        "export_dir": "./data/exports",
        "save_all_candidates": True
    }
}

# 7. Sector Banking Swing Reversion
configs["sector_banking_swing_reversion"] = {
    "name": "Sector Banking Swing Reversion",
    "description": "Swing trading mean reversion models in liquid banking stocks",
    "trading_styles": ["swing"],
    "style_overrides": {
        "swing": {
            "min_hold_bars": 10,
            "max_hold_bars": 360,
            "product_type": "CNC",
            "min_trades": 40
        }
    },
    "data": {
        "openalgo": {
            "host": "http://127.0.0.1:5000",
            "api_key": "${OPENALGO_API_KEY}",
            "source": "db"
        },
        "symbols": [
            {"symbol": "SBIN", "exchange": "NSE"},
            {"symbol": "ICICIBANK", "exchange": "NSE"},
            {"symbol": "AXISBANK", "exchange": "NSE"},
            {"symbol": "KOTAKBANK", "exchange": "NSE"},
            {"symbol": "HDFCBANK", "exchange": "NSE"}
        ],
        "timeframes": ["1h", "1d"],
        "start_date": "2021-01-01",
        "end_date": "2026-06-01",
        "train_pct": 0.70,
        "validation_pct": 0.15,
        "test_pct": 0.15
    },
    "generation": {
        "mode": "random",
        "target_count": 40000,
        "max_conditions_per_entry": 3,
        "allow_short": False,
        "indicator_categories": ["volatility", "momentum"],
        "multi_timeframe": False
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 4,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 40,
            "min_sharpe": 1.1,
            "max_drawdown": 0.18,
            "min_profit_factor": 1.35,
            "min_win_rate": 0.45,
            "min_cagr": 0.15
        },
        "validation": {
            "max_oos_sharpe_decay": 0.35,
            "min_walk_forward_consistency": 0.65,
            "monte_carlo_confidence": 0.95,
            "param_stability_tolerance": 0.20
        }
    },
    "evolution": {
        "enabled": True,
        "generations": 5,
        "population_size": 300,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
        "elitism_pct": 0.1,
        "tournament_size": 4
    },
    "ranking": {
        "mode": "robustness_first",
        "objectives": [
            {"metric": "sharpe", "weight": 0.35, "direction": "maximize"},
            {"metric": "sortino", "weight": 0.25, "direction": "maximize"},
            {"metric": "max_drawdown_pct", "weight": 0.20, "direction": "minimize"},
            {"metric": "profit_factor", "weight": 0.10, "direction": "maximize"},
            {"metric": "cagr", "weight": 0.10, "direction": "maximize"}
        ],
        "export_top_n": 15,
        "export_per_category": 3
    },
    "cost_model": {
        "commission_pct": 0.12,
        "slippage_pct": 0.03,
        "min_commission": 0.0
    },
    "output": {
        "storage": "csv",
        "base_dir": "./data/runs",
        "export_dir": "./data/exports",
        "save_all_candidates": True
    }
}

# 8. Defensive Pharma FMCG Swing
configs["defensive_pharma_fmcg_swing"] = {
    "name": "Defensive Pharma FMCG Swing",
    "description": "Low drawdown swing trading systems in defensive sectors (Pharma & FMCG)",
    "trading_styles": ["swing"],
    "style_overrides": {
        "swing": {
            "min_hold_bars": 10,
            "max_hold_bars": 360,
            "product_type": "CNC",
            "min_trades": 30
        }
    },
    "data": {
        "openalgo": {
            "host": "http://127.0.0.1:5000",
            "api_key": "${OPENALGO_API_KEY}",
            "source": "db"
        },
        "symbols": [
            {"symbol": "SUNPHARMA", "exchange": "NSE"},
            {"symbol": "CIPLA", "exchange": "NSE"},
            {"symbol": "DRREDDY", "exchange": "NSE"},
            {"symbol": "HINDUNILVR", "exchange": "NSE"},
            {"symbol": "ITC", "exchange": "NSE"}
        ],
        "timeframes": ["1d"],
        "start_date": "2021-01-01",
        "end_date": "2026-06-01",
        "train_pct": 0.70,
        "validation_pct": 0.15,
        "test_pct": 0.15
    },
    "generation": {
        "mode": "random",
        "target_count": 30000,
        "max_conditions_per_entry": 3,
        "allow_short": False,
        "indicator_categories": ["trend", "momentum", "volatility"],
        "multi_timeframe": False
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 4,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 30,
            "min_sharpe": 1.2,
            "max_drawdown": 0.12,
            "min_profit_factor": 1.40,
            "min_win_rate": 0.48,
            "min_cagr": 0.12
        },
        "validation": {
            "max_oos_sharpe_decay": 0.25,
            "min_walk_forward_consistency": 0.75,
            "monte_carlo_confidence": 0.95,
            "param_stability_tolerance": 0.15
        }
    },
    "evolution": {
        "enabled": True,
        "generations": 5,
        "population_size": 250,
        "mutation_rate": 0.25,
        "crossover_rate": 0.5,
        "elitism_pct": 0.15,
        "tournament_size": 3
    },
    "ranking": {
        "mode": "robustness_first",
        "objectives": [
            {"metric": "sharpe", "weight": 0.40, "direction": "maximize"},
            {"metric": "max_drawdown_pct", "weight": 0.30, "direction": "minimize"},
            {"metric": "sortino", "weight": 0.20, "direction": "maximize"},
            {"metric": "profit_factor", "weight": 0.10, "direction": "maximize"}
        ],
        "export_top_n": 15,
        "export_per_category": 3
    },
    "cost_model": {
        "commission_pct": 0.12,
        "slippage_pct": 0.02,
        "min_commission": 0.0
    },
    "output": {
        "storage": "csv",
        "base_dir": "./data/runs",
        "export_dir": "./data/exports",
        "save_all_candidates": True
    }
}

# 9. NSE Index Options Volatility Intraday
configs["nse_index_options_volatility_intraday"] = {
    "name": "NSE Index Options Volatility Intraday",
    "description": "Volatility breakout and expansion models for Index Option buyers/sellers on NIFTY & BANKNIFTY",
    "trading_styles": ["intraday"],
    "style_overrides": {
        "intraday": {
            "max_hold_bars": 75,
            "forced_exit_time": "15:15",
            "product_type": "MIS",
            "min_trades": 200
        }
    },
    "data": {
        "openalgo": {
            "host": "http://127.0.0.1:5000",
            "api_key": "${OPENALGO_API_KEY}",
            "source": "db"
        },
        "symbols": [
            {"symbol": "NIFTY", "exchange": "NSE"},
            {"symbol": "BANKNIFTY", "exchange": "NSE"}
        ],
        "timeframes": ["5m", "15m"],
        "start_date": "2021-01-01",
        "end_date": "2026-06-01",
        "train_pct": 0.70,
        "validation_pct": 0.15,
        "test_pct": 0.15
    },
    "generation": {
        "mode": "guided",
        "target_count": 50000,
        "max_conditions_per_entry": 3,
        "allow_short": True,
        "indicator_categories": ["volatility", "volume", "momentum"],
        "multi_timeframe": True
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 4,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 200,
            "min_sharpe": 1.3,
            "max_drawdown": 0.25,
            "min_profit_factor": 1.45,
            "min_win_rate": 0.40,
            "min_cagr": 0.20
        },
        "validation": {
            "max_oos_sharpe_decay": 0.35,
            "min_walk_forward_consistency": 0.65,
            "monte_carlo_confidence": 0.95,
            "param_stability_tolerance": 0.20
        }
    },
    "evolution": {
        "enabled": True,
        "generations": 6,
        "population_size": 400,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
        "elitism_pct": 0.1,
        "tournament_size": 5
    },
    "ranking": {
        "mode": "robustness_first",
        "objectives": [
            {"metric": "sharpe", "weight": 0.30, "direction": "maximize"},
            {"metric": "sortino", "weight": 0.20, "direction": "maximize"},
            {"metric": "max_drawdown_pct", "weight": 0.25, "direction": "minimize"},
            {"metric": "profit_factor", "weight": 0.15, "direction": "maximize"},
            {"metric": "cagr", "weight": 0.10, "direction": "maximize"}
        ],
        "export_top_n": 20,
        "export_per_category": 4
    },
    "cost_model": {
        "commission_pct": 0.04,
        "slippage_pct": 0.03,
        "min_commission": 20.0
    },
    "output": {
        "storage": "csv",
        "base_dir": "./data/runs",
        "export_dir": "./data/exports",
        "save_all_candidates": True
    }
}

# 10. Nifty Multi-Timeframe Robust
configs["nifty_multi_timeframe_robust"] = {
    "name": "Nifty Multi-Timeframe Robust",
    "description": "Ultra-stable multi-timeframe swing and intraday strategy discovery for NIFTY 50",
    "trading_styles": ["intraday", "swing"],
    "style_overrides": {
        "intraday": {
            "max_hold_bars": 75,
            "forced_exit_time": "15:15",
            "product_type": "MIS",
            "min_trades": 150
        },
        "swing": {
            "min_hold_bars": 10,
            "max_hold_bars": 360,
            "product_type": "CNC",
            "min_trades": 40
        }
    },
    "data": {
        "openalgo": {
            "host": "http://127.0.0.1:5000",
            "api_key": "${OPENALGO_API_KEY}",
            "source": "db"
        },
        "symbols": [
            {"symbol": "NIFTY", "exchange": "NSE"}
        ],
        "timeframes": ["5m", "15m", "1h", "1d"],
        "start_date": "2020-01-01",
        "end_date": "2026-06-01",
        "train_pct": 0.70,
        "validation_pct": 0.15,
        "test_pct": 0.15
    },
    "generation": {
        "mode": "guided",
        "target_count": 60000,
        "max_conditions_per_entry": 4,
        "allow_short": True,
        "indicator_categories": ["trend", "momentum", "volatility", "volume"],
        "multi_timeframe": True
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 5,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 50,
            "min_sharpe": 1.2,
            "max_drawdown": 0.18,
            "min_profit_factor": 1.45,
            "min_win_rate": 0.42,
            "min_cagr": 0.18
        },
        "validation": {
            "max_oos_sharpe_decay": 0.30,
            "min_walk_forward_consistency": 0.70,
            "monte_carlo_confidence": 0.95,
            "param_stability_tolerance": 0.18
        }
    },
    "evolution": {
        "enabled": True,
        "generations": 7,
        "population_size": 400,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
        "elitism_pct": 0.1,
        "tournament_size": 5
    },
    "ranking": {
        "mode": "robustness_first",
        "objectives": [
            {"metric": "sharpe", "weight": 0.30, "direction": "maximize"},
            {"metric": "sortino", "weight": 0.20, "direction": "maximize"},
            {"metric": "max_drawdown_pct", "weight": 0.25, "direction": "minimize"},
            {"metric": "profit_factor", "weight": 0.15, "direction": "maximize"},
            {"metric": "cagr", "weight": 0.10, "direction": "maximize"}
        ],
        "export_top_n": 20,
        "export_per_category": 5
    },
    "cost_model": {
        "commission_pct": 0.05,
        "slippage_pct": 0.02,
        "min_commission": 10.0
    },
    "output": {
        "storage": "csv",
        "base_dir": "./data/runs",
        "export_dir": "./data/exports",
        "save_all_candidates": True
    }
}

# Write YAML files
config_dir = Path("config")
config_dir.mkdir(exist_ok=True)

success_count = 0
for name, data in configs.items():
    file_path = config_dir / f"{name}.yaml"
    with open(file_path, "w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    
    # Try validating it
    try:
        loaded = load_config(file_path)
        print(f"Validated and saved: {file_path}")
        success_count += 1
    except Exception as e:
        print(f"Failed validation for {file_path}: {e}")

print(f"\nSuccessfully generated and validated {success_count} config files!")
