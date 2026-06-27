import yaml
from pathlib import Path
from quant_engine.config import load_config

# Define 10 new configurations focused primarily on intraday setups in the Indian market
configs = {}

# 1. Nifty ORB (Opening Range Breakout) Intraday
configs["nifty_orb_breakout_intraday"] = {
    "name": "Nifty ORB Breakout Intraday",
    "description": "Exploit early morning momentum and range breakouts on the NIFTY 50 index",
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
        "target_count": 40000,
        "max_conditions_per_entry": 3,
        "allow_short": True,
        "indicator_categories": ["trend", "volatility"],
        "multi_timeframe": False
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 4,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 200,
            "min_sharpe": 1.2,
            "max_drawdown": 0.18,
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
        "population_size": 350,
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

# 2. Sector Auto Momentum Intraday
configs["sector_auto_momentum_intraday"] = {
    "name": "Sector Auto Momentum Intraday",
    "description": "High momentum intraday strategy for liquid Auto stocks (TATAMOTORS, M&M, MARUTI)",
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
            {"symbol": "TATAMOTORS", "exchange": "NSE"},
            {"symbol": "M&M", "exchange": "NSE"},
            {"symbol": "MARUTI", "exchange": "NSE"}
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
        "indicator_categories": ["momentum", "trend", "volume"],
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
            "max_drawdown": 0.20,
            "min_profit_factor": 1.35,
            "min_win_rate": 0.38,
            "min_cagr": 0.15
        },
        "validation": {
            "max_oos_sharpe_decay": 0.35,
            "min_walk_forward_consistency": 0.60,
            "monte_carlo_confidence": 0.95,
            "param_stability_tolerance": 0.25
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

# 3. Sector Metal Volatility Intraday
configs["sector_metal_volatility_intraday"] = {
    "name": "Sector Metal Volatility Intraday",
    "description": "Volatility expansion breakout setups in high-beta metal stocks (TATASTEEL, JSWSTEEL, HINDALCO)",
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
            {"symbol": "TATASTEEL", "exchange": "NSE"},
            {"symbol": "JSWSTEEL", "exchange": "NSE"},
            {"symbol": "HINDALCO", "exchange": "NSE"}
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
        "target_count": 35000,
        "max_conditions_per_entry": 3,
        "allow_short": True,
        "indicator_categories": ["volatility", "momentum", "volume"],
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
            "min_win_rate": 0.40,
            "min_cagr": 0.15
        },
        "validation": {
            "max_oos_sharpe_decay": 0.35,
            "min_walk_forward_consistency": 0.60,
            "monte_carlo_confidence": 0.95,
            "param_stability_tolerance": 0.25
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

# 4. PSU Bank High Beta Intraday
configs["psu_bank_high_beta_intraday"] = {
    "name": "PSU Bank High Beta Intraday",
    "description": "High beta intraday momentum on public-sector banking giants (SBIN, BANKBARODA, CANBK)",
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
            {"symbol": "SBIN", "exchange": "NSE"},
            {"symbol": "BANKBARODA", "exchange": "NSE"},
            {"symbol": "CANBK", "exchange": "NSE"}
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
            "min_trades": 200,
            "min_sharpe": 1.2,
            "max_drawdown": 0.18,
            "min_profit_factor": 1.4,
            "min_win_rate": 0.42,
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

# 5. Reliance VWAP Scalp Intraday
configs["reliance_vwap_scalp_intraday"] = {
    "name": "Reliance VWAP Scalp Intraday",
    "description": "VWAP and Volume SMA breakout scalping on India's heaviest stock, RELIANCE",
    "trading_styles": ["intraday"],
    "style_overrides": {
        "intraday": {
            "max_hold_bars": 75,
            "forced_exit_time": "15:15",
            "product_type": "MIS",
            "min_trades": 250
        }
    },
    "data": {
        "openalgo": {
            "host": "http://127.0.0.1:5000",
            "api_key": "${OPENALGO_API_KEY}",
            "source": "db"
        },
        "symbols": [
            {"symbol": "RELIANCE", "exchange": "NSE"}
        ],
        "timeframes": ["5m"],
        "start_date": "2021-01-01",
        "end_date": "2026-06-01",
        "train_pct": 0.70,
        "validation_pct": 0.15,
        "test_pct": 0.15
    },
    "generation": {
        "mode": "guided",
        "target_count": 40000,
        "max_conditions_per_entry": 3,
        "allow_short": True,
        "indicator_categories": ["volume", "trend", "momentum"],
        "multi_timeframe": False
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 4,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 250,
            "min_sharpe": 1.2,
            "max_drawdown": 0.15,
            "min_profit_factor": 1.4,
            "min_win_rate": 0.45,
            "min_cagr": 0.15
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
        "generations": 6,
        "population_size": 350,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
        "elitism_pct": 0.1,
        "tournament_size": 5
    },
    "ranking": {
        "mode": "robustness_first",
        "objectives": [
            {"metric": "sharpe", "weight": 0.40, "direction": "maximize"},
            {"metric": "profit_factor", "weight": 0.20, "direction": "maximize"},
            {"metric": "max_drawdown_pct", "weight": 0.20, "direction": "minimize"},
            {"metric": "cagr", "weight": 0.20, "direction": "maximize"}
        ],
        "export_top_n": 15,
        "export_per_category": 3
    },
    "cost_model": {
        "commission_pct": 0.03,
        "slippage_pct": 0.015,
        "min_commission": 20.0
    },
    "output": {
        "storage": "csv",
        "base_dir": "./data/runs",
        "export_dir": "./data/exports",
        "save_all_candidates": True
    }
}

# 6. Nifty Realty Infra Momentum Intraday
configs["nifty_realty_infra_momentum_intraday"] = {
    "name": "Nifty Realty Infra Momentum Intraday",
    "description": "High beta intraday momentum in Realty and Infra sectors (DLF, L&T, ADANIPORTS)",
    "trading_styles": ["intraday"],
    "style_overrides": {
        "intraday": {
            "max_hold_bars": 75,
            "forced_exit_time": "15:15",
            "product_type": "MIS",
            "min_trades": 160
        }
    },
    "data": {
        "openalgo": {
            "host": "http://127.0.0.1:5000",
            "api_key": "${OPENALGO_API_KEY}",
            "source": "db"
        },
        "symbols": [
            {"symbol": "DLF", "exchange": "NSE"},
            {"symbol": "LT", "exchange": "NSE"},
            {"symbol": "ADANIPORTS", "exchange": "NSE"}
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
        "target_count": 35000,
        "max_conditions_per_entry": 3,
        "allow_short": True,
        "indicator_categories": ["trend", "momentum", "volume"],
        "multi_timeframe": False
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 4,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 160,
            "min_sharpe": 1.1,
            "max_drawdown": 0.22,
            "min_profit_factor": 1.35,
            "min_win_rate": 0.38,
            "min_cagr": 0.15
        },
        "validation": {
            "max_oos_sharpe_decay": 0.35,
            "min_walk_forward_consistency": 0.60,
            "monte_carlo_confidence": 0.95,
            "param_stability_tolerance": 0.25
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

# 7. Consumer FMCG Low Vol Intraday
configs["consumer_fmcg_low_vol_intraday"] = {
    "name": "Consumer FMCG Low Vol Intraday",
    "description": "Steady intraday trend-following in defensive, low-volatility FMCG stocks (HINDUNILVR, ITC, BRITANNIA)",
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
            {"symbol": "HINDUNILVR", "exchange": "NSE"},
            {"symbol": "ITC", "exchange": "NSE"},
            {"symbol": "BRITANNIA", "exchange": "NSE"}
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
            "min_sharpe": 1.1,
            "max_drawdown": 0.12,
            "min_profit_factor": 1.35,
            "min_win_rate": 0.45,
            "min_cagr": 0.10
        },
        "validation": {
            "max_oos_sharpe_decay": 0.25,
            "min_walk_forward_consistency": 0.70,
            "monte_carlo_confidence": 0.95,
            "param_stability_tolerance": 0.18
        }
    },
    "evolution": {
        "enabled": True,
        "generations": 5,
        "population_size": 250,
        "mutation_rate": 0.25,
        "crossover_rate": 0.5,
        "elitism_pct": 0.12,
        "tournament_size": 3
    },
    "ranking": {
        "mode": "robustness_first",
        "objectives": [
            {"metric": "sharpe", "weight": 0.40, "direction": "maximize"},
            {"metric": "max_drawdown_pct", "weight": 0.30, "direction": "minimize"},
            {"metric": "profit_factor", "weight": 0.20, "direction": "maximize"},
            {"metric": "cagr", "weight": 0.10, "direction": "maximize"}
        ],
        "export_top_n": 15,
        "export_per_category": 3
    },
    "cost_model": {
        "commission_pct": 0.03,
        "slippage_pct": 0.015,
        "min_commission": 20.0
    },
    "output": {
        "storage": "csv",
        "base_dir": "./data/runs",
        "export_dir": "./data/exports",
        "save_all_candidates": True
    }
}

# 8. Heavyweight Short Scalper Intraday
configs["heavyweight_short_scalper_intraday"] = {
    "name": "Heavyweight Short Scalper Intraday",
    "description": "Short-only intraday scalping models for heavyweights (INFOSYS, HDFCBANK, ICICIBANK) to profit on down-days",
    "trading_styles": ["intraday"],
    "style_overrides": {
        "intraday": {
            "max_hold_bars": 150,
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
            {"symbol": "INFOSYS", "exchange": "NSE"},
            {"symbol": "HDFCBANK", "exchange": "NSE"},
            {"symbol": "ICICIBANK", "exchange": "NSE"}
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
        "target_count": 40000,
        "max_conditions_per_entry": 3,
        "allow_short": True,  # Generate short setups
        "indicator_categories": ["trend", "momentum", "volatility"],
        "multi_timeframe": False
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 4,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 200,
            "min_sharpe": 1.1,
            "max_drawdown": 0.18,
            "min_profit_factor": 1.35,
            "min_win_rate": 0.40,
            "min_cagr": 0.12
        },
        "validation": {
            "max_oos_sharpe_decay": 0.35,
            "min_walk_forward_consistency": 0.60,
            "monte_carlo_confidence": 0.95,
            "param_stability_tolerance": 0.22
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
            {"metric": "sortino", "weight": 0.20, "direction": "maximize"},
            {"metric": "max_drawdown_pct", "weight": 0.25, "direction": "minimize"},
            {"metric": "profit_factor", "weight": 0.20, "direction": "maximize"}
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

# 9. Nifty Energy Power Trend Intraday
configs["nifty_energy_power_trend_intraday"] = {
    "name": "Nifty Energy Power Trend Intraday",
    "description": "Trend-following intraday strategies for heavy volume energy sector stocks (NTPC, POWERGRID, ONGC)",
    "trading_styles": ["intraday"],
    "style_overrides": {
        "intraday": {
            "max_hold_bars": 75,
            "forced_exit_time": "15:15",
            "product_type": "MIS",
            "min_trades": 160
        }
    },
    "data": {
        "openalgo": {
            "host": "http://127.0.0.1:5000",
            "api_key": "${OPENALGO_API_KEY}",
            "source": "db"
        },
        "symbols": [
            {"symbol": "NTPC", "exchange": "NSE"},
            {"symbol": "POWERGRID", "exchange": "NSE"},
            {"symbol": "ONGC", "exchange": "NSE"}
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
        "indicator_categories": ["trend", "volume"],
        "multi_timeframe": False
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 4,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 160,
            "min_sharpe": 1.1,
            "max_drawdown": 0.18,
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

# 10. Multi-Symbol High Frequency Reversion Intraday
configs["multisymbol_high_frequency_reversion_intraday"] = {
    "name": "Multi-Symbol High Frequency Reversion Intraday",
    "description": "High-frequency intraday mean reversion across the top 5 liquid stock heavyweights on 5m charts",
    "trading_styles": ["intraday"],
    "style_overrides": {
        "intraday": {
            "max_hold_bars": 150,
            "forced_exit_time": "15:15",
            "product_type": "MIS",
            "min_trades": 300
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
            {"symbol": "ICICIBANK", "exchange": "NSE"}
        ],
        "timeframes": ["5m"],
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
        "indicator_categories": ["volatility", "momentum"],
        "multi_timeframe": False
    },
    "filters": {
        "fast_reject": {
            "max_complexity": 4,
            "min_indicators": 1
        },
        "backtest": {
            "min_trades": 300,
            "min_sharpe": 1.2,
            "max_drawdown": 0.15,
            "min_profit_factor": 1.40,
            "min_win_rate": 0.45,
            "min_cagr": 0.15
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
        "generations": 6,
        "population_size": 400,
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
            {"metric": "max_drawdown_pct", "weight": 0.25, "direction": "minimize"},
            {"metric": "profit_factor", "weight": 0.20, "direction": "maximize"}
        ],
        "export_top_n": 20,
        "export_per_category": 4
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
