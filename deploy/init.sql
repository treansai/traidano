CREATE TABLE bots (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255),
    market VARCHAR(255) NOT NULL,
    trading_strategy VARCHAR(255) NOT NULL,
    symbols TEXT NOT NULL,
    lookback INT NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    risk_per_trade DOUBLE PRECISION NOT NULL,
    max_positions INT NOT NULL,
    timeframes TEXT NOT NULL,
    volatility_window INT NOT NULL,
    volatility_threshold DOUBLE PRECISION NOT NULL,
    is_running BOOLEAN DEFAULT FALSE
);
-- zflub$0$1$9$8$!'aafg79ydhkhdfqagk65a6kgd12'

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    bot_id TEXT NOT NULL REFERENCES bots(id),
    token_address TEXT NOT NULL,
    token_symbol TEXT NOT NULL,
    amount NUMERIC(24, 12) NOT NULL,
    price_usd NUMERIC(24, 12) NOT NULL,
    transaction_signature TEXT,
    transaction_type TEXT NOT NULL CHECK (transaction_type IN ('buy', 'sell')),
    profit_loss NUMERIC(24, 12),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS wallets (
    id SERIAL PRIMARY KEY,
    bot_id TEXT NOT NULL REFERENCES bots(id),
    public_key TEXT NOT NULL,
    encrypted_private_key TEXT,
    label TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_bot_wallet UNIQUE (bot_id, public_key)
);