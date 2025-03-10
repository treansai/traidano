use crate::base::AppState;
use crate::bot::strategies::should_execute;
use crate::bot::{BotConfig, MarketType};
use crate::core::functions::calculate_position_size;
use crate::dao::solana::{record_memecoin_trade, get_bot_solana_wallet};
use crate::models::order::{Order, Qty};
use crate::models::trade::{Side, TimeInForce, Type};
use axum::extract::State;
use axum::Json;
use opentelemetry::KeyValue;
use reqwest::{Client as ReqwestClient, header};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;
use anyhow::{Result, Error as AnyhowError, Context};
use solana_client::rpc_client::RpcClient;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    pubkey::Pubkey,
    signature::{Keypair, Signer},
    system_instruction,
    transaction::Transaction,
};
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};
use chrono::{DateTime, Utc};
use rand::Rng;
use base64;
use bincode;

// Constants
// const LEARNING_RATE: f64 = 0.0005;
// const DISCOUNT_FACTOR: f64 = 0.95;
// const EPSILON: f64 = 0.15;
// const MAX_POSITION_SIZE: f64 = 0.05;
// const TRADING_FEE: f64 = 0.003;
// const SLIPPAGE_TOLERANCE: f64 = 0.05;
// const RPC_URL: &str = "https://api.mainnet-beta.solana.com";
// const JUPITER_QUOTE_API: &str = "https://quote-api.jup.ag/v6/quote";
// const JUPITER_SWAP_API: &str = "https://quote-api.jup.ag/v6/swap";
// const SOL_MINT_ADDRESS: &str = "So11111111111111111111111111111111111111112";
// const USDC_MINT_ADDRESS: &str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";
const STATE_HISTORY_SIZE: usize = 20;

pub struct RLConfig {
    learning_rate: f32,
    discount_factor: f32,
    epsilon: f32,
    max_position_size: f32,
    trading_fee: f32,
    slippage_tolerance: f32,
    rpc_url: String,
    jupiter_quote_api: String,
    jupiter_swap_api: String,
    sol_mint_address: String,
    usdc_mint_address: String,
    state_history_size: usize,
}

// Trading State
#[derive(Debug, Clone)]
struct TradingState {
    sol_balance: f64,
    token_balance: f64,
    price_history: Vec<f64>,
    volume_history: Vec<f64>,
    sentiment_history: Vec<f64>,
    position_value: f64,
    token_mint: String,
}

impl Default for TradingState {
    fn default() -> Self {
        Self {
            sol_balance: 0.0,
            token_balance: 0.0,
            price_history: Vec::with_capacity(STATE_HISTORY_SIZE),
            volume_history: Vec::with_capacity(STATE_HISTORY_SIZE),
            sentiment_history: Vec::with_capacity(STATE_HISTORY_SIZE),
            position_value: 0.0,
            token_mint: String::new(),
        }
    }
}

// Q-Network for Reinforcement Learning
struct QNetwork {
    model: nn::Sequential
}

impl QNetwork {
    fn new(vs: &nn::Path) -> Self {
        let model = nn::seq()
            .add(nn::linear(vs / "layer1", 20, 128, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "layer2", 128, 256, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "layer3", 256, 128, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "layer4", 128, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "layer5", 64, 3, Default::default()));
        Self { model }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        self.model.forward(xs)
    }
}

// Jupiter API types
#[derive(Debug, Serialize, Deserialize)]
struct JupiterQuoteRequest {
    input_mint: String,
    output_mint: String,
    amount: String,
    slippage_bps: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct JupiterQuoteResponse {
    #[serde(rename = "inputMint")]
    input_mint: String,
    #[serde(rename = "outputMint")]
    output_mint: String,
    #[serde(rename = "inAmount")]
    in_amount: String,
    #[serde(rename = "outAmount")]
    out_amount: String,
    #[serde(rename = "otherAmountThreshold")]
    other_amount_threshold: String,
    #[serde(rename = "swapMode")]
    swap_mode: String,
    #[serde(rename = "slippageBps")]
    slippage_bps: u32,
    routes: Vec<JupiterRoute>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JupiterRoute {
    #[serde(rename = "marketInfos")]
    market_infos: Vec<JupiterMarketInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JupiterMarketInfo {
    id: String,
    label: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct JupiterSwapRequest {
    route: JupiterQuoteResponse,
    #[serde(rename = "userPublicKey")]
    user_public_key: String,
    #[serde(rename = "wrapAndUnwrapSol")]
    wrap_and_unwrap_sol: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct JupiterSwapResponse {
    #[serde(rename = "swapTransaction")]
    swap_transaction: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct TokenMetadata {
    symbol: String,
    name: String,
    address: String,
    #[serde(rename = "logoURI", default)]
    logo_uri: Option<String>,
    #[serde(rename = "coingeckoId", default)]
    coingecko_id: Option<String>,
    decimals: u8,
}

// Trading agent that combines RL with execution
struct RLTradingAgent {
    state: TradingState,
    q_network: QNetwork,
    optimizer: nn::Optimizer,
    device: Device,
    http_client: ReqwestClient,
    rpc_client: RpcClient,
    wallet_keypair: Option<Keypair>,
    wallet_public_key: String,
    bot_id: String,
    app_state: Arc<AppState>,
    vs: nn::VarStore,
}

impl RLTradingAgent {
    fn new(app_state: Arc<AppState>, bot_id: String, wallet_public_key: String, token_mint: String) -> Result<Self, AnyhowError> {
        let vs = nn::VarStore::new(Device::Cpu);
        let q_network = QNetwork::new(&vs.root());
        let optimizer = nn::Adam::default().build(&vs, LEARNING_RATE).unwrap();

        let state = TradingState {
            sol_balance: 0.0,
            token_balance: 0.0,
            price_history: Vec::new(),
            volume_history: Vec::new(),
            sentiment_history: Vec::new(),
            position_value: 0.0,
            token_mint,
        };

        let http_client = create_http_client();
        let rpc_client = RpcClient::new_with_commitment(
            RPC_URL.to_string(),
            CommitmentConfig::confirmed(),
        );

        // In a real implementation, you'd securely retrieve the keypair
        // Here we're just storing the public key
        let wallet_keypair = None;

        Ok(Self {
            state,
            q_network,
            optimizer,
            device: Device::Cpu,
            http_client,
            rpc_client,
            wallet_keypair,
            wallet_public_key,
            bot_id,
            app_state,
            vs,
        })
    }

    fn get_state_tensor(&self) -> Tensor {
        // Check if we have enough data
        if self.state.price_history.len() < STATE_HISTORY_SIZE || 
           self.state.volume_history.len() < STATE_HISTORY_SIZE || 
           self.state.sentiment_history.len() < STATE_HISTORY_SIZE {
            // Return zeros if not enough data
            return Tensor::zeros(&[1, 20], (tch::Kind::Float, self.device));
        }

        let price_data: Vec<f32> = self.state.price_history.iter().map(|p| *p as f32).collect();
        let volume_data: Vec<f32> = self.state.volume_history.iter().map(|v| *v as f32).collect();
        let sentiment_data: Vec<f32> = self.state.sentiment_history.iter().map(|s| *s as f32).collect();

        let mut state_data = Vec::new();
        state_data.extend(price_data);
        state_data.extend(volume_data);
        state_data.extend(sentiment_data);

        // Normalize data
        let mean = state_data.iter().sum::<f32>() / state_data.len() as f32;
        let std_dev = (state_data.iter().map(|x| (*x - mean).powi(2)).sum::<f32>() / state_data.len() as f32).sqrt();
        let normalized: Vec<f32> = if std_dev > 1e-6 {
            state_data.iter().map(|x| (*x - mean) / std_dev).collect()
        } else {
            state_data
        };

        Tensor::of_slice(&normalized)
            .view([-1, STATE_HISTORY_SIZE * 3])
            .to_device(self.device)
    }

    fn select_action(&self, state: &Tensor, epsilon: f64) -> i64 {
        let mut rng = rand::thread_rng();
        
        // Exploration: random action with probability epsilon
        if rng.gen::<f64>() < epsilon {
            return rng.gen_range(-1..=1);
        } 
        
        // Exploitation: choose best action based on Q-values
        let q_values = self.q_network.forward(state);
        let best_action_idx = q_values.argmax(1, false).int64_value(&[0]);
        best_action_idx - 1 // Convert [0,1,2] to [-1,0,1]
    }

    async fn fetch_jupiter_quote(&self, input_mint: &str, output_mint: &str, amount: u64) -> Result<f64, AnyhowError> {
        let url = format!(
            "{}?inputMint={}&outputMint={}&amount={}&slippageBps={}",
            JUPITER_QUOTE_API,
            input_mint,
            output_mint,
            amount,
            (SLIPPAGE_TOLERANCE * 10000.0) as u64 // Convert to basis points
        );

        let response = self.http_client.get(&url).send().await
            .context("Failed to send Jupiter quote request")?;
        let json: Value = response.json().await
            .context("Failed to parse Jupiter quote response")?;

        let out_amount_str = json["outAmount"].as_str()
            .ok_or_else(|| AnyhowError::msg("No outAmount in Jupiter response"))?;
        let out_amount = out_amount_str.parse::<u64>()
            .context("Failed to parse outAmount as u64")?;
        
        let in_amount = amount as f64;
        let price = out_amount as f64 / in_amount;
        
        Ok(price)
    }

    async fn fetch_market_data(&mut self) -> Result<(), AnyhowError> {
        // Fetch latest price
        let new_price = self.fetch_jupiter_quote(
            SOL_MINT_ADDRESS, 
            &self.state.token_mint, 
            1_000_000_000 // 1 SOL in lamports
        ).await?;
        
        // In a real implementation, you'd fetch the actual volume
        // For now we'll simulate volume with a random value influenced by price changes
        let price_change = if !self.state.price_history.is_empty() {
            let last_price = self.state.price_history.last().unwrap();
            (new_price - last_price) / last_price
        } else {
            0.0
        };
        
        let volatility_factor = 1.0 + price_change.abs() * 10.0;
        let new_volume = rand::thread_rng().gen_range(1000.0..10000.0) * volatility_factor;
        
        // Calculate sentiment score based on volume and price movement
        let avg_volume = if !self.state.volume_history.is_empty() {
            self.state.volume_history.iter().sum::<f64>() / self.state.volume_history.len() as f64
        } else {
            new_volume
        };
        
        let volume_sentiment = if new_volume > avg_volume * 1.5 { 1.0 } 
                         else if new_volume < avg_volume * 0.5 { -0.5 } 
                         else { 0.0 };
        
        let price_sentiment = if price_change > 0.03 { 1.0 }
                        else if price_change < -0.03 { -1.0 }
                        else { 0.0 };
        
        let sentiment = (volume_sentiment + price_sentiment) / 2.0;
        
        // Add to history
        self.state.price_history.push(new_price);
        self.state.volume_history.push(new_volume);
        self.state.sentiment_history.push(sentiment);
        
        // Maintain fixed window size
        if self.state.price_history.len() > STATE_HISTORY_SIZE {
            self.state.price_history.remove(0);
        }
        if self.state.volume_history.len() > STATE_HISTORY_SIZE {
            self.state.volume_history.remove(0);
        }
        if self.state.sentiment_history.len() > STATE_HISTORY_SIZE {
            self.state.sentiment_history.remove(0);
        }
        
        tracing::debug!(
            "Market data updated: price={}, volume={}, sentiment={}", 
            new_price, new_volume, sentiment
        );
        
        Ok(())
    }

    async fn simulate_trade(&mut self, action: i64, position_size: f64) -> Result<f64, AnyhowError> {
        // Get current balances and price
        let current_price = if let Some(price) = self.state.price_history.last() {
            *price
        } else {
            self.fetch_jupiter_quote(SOL_MINT_ADDRESS, &self.state.token_mint, 1_000_000_000).await?
        };
        
        let old_portfolio_value = self.state.sol_balance + (self.state.token_balance * current_price);
        
        // Record the action for metrics
        let action_type = match action {
            1 => "buy",
            -1 => "sell",
            _ => "hold",
        };
        
        tracing::info!(
            "RL agent action: {}, position_size: {}, current_price: {}", 
            action_type, position_size, current_price
        );
        
        match action {
            1 => { // Buy SOL -> Token
                if self.state.sol_balance > 0.0 {
                    let sol_to_spend = self.state.sol_balance * position_size;
                    let tokens_bought = (sol_to_spend / current_price) * (1.0 - TRADING_FEE);
                    
                    self.state.sol_balance -= sol_to_spend;
                    self.state.token_balance += tokens_bought;
                    
                    // Record the trade in the database
                    if let Err(e) = record_memecoin_trade(
                        &self.app_state.db,
                        &self.bot_id,
                        &self.state.token_mint,
                        "TOKEN_SYMBOL", // In a real implementation, get the actual symbol
                        tokens_bought,
                        current_price,
                        None, // Transaction signature
                        "buy",
                        None, // Profit/loss
                    ).await {
                        tracing::error!("Failed to record buy trade: {:?}", e);
                    }
                    
                    tracing::info!(
                        "Bought {} tokens with {} SOL at price {}", 
                        tokens_bought, sol_to_spend, current_price
                    );
                }
            },
            -1 => { // Sell Token -> SOL
                if self.state.token_balance > 0.0 {
                    let tokens_to_sell = self.state.token_balance * position_size;
                    let sol_received = (tokens_to_sell * current_price) * (1.0 - TRADING_FEE);
                    
                    self.state.token_balance -= tokens_to_sell;
                    self.state.sol_balance += sol_received;
                    
                    let profit_loss = sol_received - (tokens_to_sell * current_price);
                    
                    // Record the trade in the database
                    if let Err(e) = record_memecoin_trade(
                        &self.app_state.db,
                        &self.bot_id,
                        &self.state.token_mint,
                        "TOKEN_SYMBOL", // In a real implementation, get the actual symbol
                        tokens_to_sell,
                        current_price,
                        None, // Transaction signature
                        "sell",
                        Some(profit_loss),
                    ).await {
                        tracing::error!("Failed to record sell trade: {:?}", e);
                    }
                    
                    tracing::info!(
                        "Sold {} tokens for {} SOL at price {}", 
                        tokens_to_sell, sol_received, current_price
                    );
                }
            },
            _ => {} // Hold - do nothing
        }
        
        // Calculate reward based on portfolio value change
        let new_portfolio_value = self.state.sol_balance + (self.state.token_balance * current_price);
        let value_change = new_portfolio_value - old_portfolio_value;
        let reward = value_change / old_portfolio_value;
        
        // Add sentiment influence to reward
        if let Some(sentiment) = self.state.sentiment_history.last() {
            let sentiment_reward = *sentiment * 0.1;
            let total_reward = reward + sentiment_reward;
            
            // Update position value
            self.state.position_value = new_portfolio_value;
            
            tracing::debug!(
                "Reward: value_change={}, sentiment_contribution={}, total={}", 
                reward, sentiment_reward, total_reward
            );
            
            return Ok(total_reward);
        }
        
        self.state.position_value = new_portfolio_value;
        Ok(reward)
    }

    async fn train_step(&mut self) -> Result<(), AnyhowError> {
        // Fetch latest market data
        self.fetch_market_data().await?;
        
        // Get current state
        let state_tensor = self.get_state_tensor();
        
        // Select action
        let action = self.select_action(&state_tensor, EPSILON);
        
        // Execute trade and get reward
        let reward = self.simulate_trade(action, MAX_POSITION_SIZE).await?;
        
        // Get new state after trade
        let next_state_tensor = self.get_state_tensor();
        
        // Calculate target Q-value with temporal difference
        let q_values = self.q_network.forward(&state_tensor);
        let next_q_values = self.q_network.forward(&next_state_tensor);
        
        let max_next_q = next_q_values.max_dim(1, false).0;
        
        let reward_tensor = Tensor::of_slice(&[reward as f32])
            .to_device(self.device);
        let discount_tensor = Tensor::of_slice(&[DISCOUNT_FACTOR as f32])
            .to_device(self.device);
        
        let target = reward_tensor + discount_tensor * max_next_q;
        
        // Get the Q-value for the taken action
        let action_idx = (action + 1) as i64; // Convert to [0,1,2]
        let action_tensor = Tensor::of_slice(&[action_idx])
            .to_device(self.device)
            .unsqueeze(0);
        
        let predicted_q = q_values.gather(1, &action_tensor, false).squeeze();
        
        // Compute loss
        let loss = predicted_q.mse_loss(&target, tch::Reduction::Mean);
        
        // Update network
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();
        
        tracing::debug!(
            "Training step - action: {}, reward: {}, loss: {}", 
            action, reward, f32::from(&loss)
        );
        
        Ok(())
    }

    async fn update_portfolio_from_onchain(&mut self) -> Result<(), AnyhowError> {
        // In a real implementation, you would:
        // 1. Query token account balance for self.state.token_mint
        // 2. Query SOL balance
        // 3. Update self.state.sol_balance and self.state.token_balance
        
        // For simulation, we'll just set some values if they're zero
        if self.state.sol_balance == 0.0 && self.state.token_balance == 0.0 {
            self.state.sol_balance = 1.0; // 1 SOL
            tracing::info!("Initialized portfolio with 1 SOL for simulation");
        }
        
        Ok(())
    }

    async fn save_model(&self) -> Result<(), AnyhowError> {
        let path = format!("models/rl_memecoin_{}.pt", self.bot_id);
        self.vs.save(&path)?;
        tracing::info!("Model saved to {}", path);
        Ok(())
    }

    async fn load_model(&mut self) -> Result<(), AnyhowError> {
        let path = format!("models/rl_memecoin_{}.pt", self.bot_id);
        match self.vs.load(&path) {
            Ok(_) => {
                tracing::info!("Model loaded from {}", path);
                Ok(())
            },
            Err(e) => {
                tracing::warn!("Could not load model, starting fresh: {:?}", e);
                Ok(()) // Not a critical error
            }
        }
    }
}

// Function to create HTTP client with rate limiting
fn create_http_client() -> ReqwestClient {
    let mut headers = header::HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    
    reqwest::Client::builder()
        .default_headers(headers)
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client")
}

// The main memecoin trading strategy function
pub async fn rl_solana_memecoin_strategy(state: Arc<AppState>, config: BotConfig) {
    let mut interval = interval(Duration::from_secs(60)); // Check every minute
    
    // Metrics
    let trade_count_counter = state.meter
        .u64_counter("rl_solana_memecoin_trades")
        .with_description("Number of RL-based memecoin trades executed")
        .init();
    
    let profit_gauge = state.meter
        .f64_gauge("rl_solana_memecoin_profit")
        .with_description("Profit/loss from RL-based memecoin trading")
        .init();
    
    // Get wallet from database or environment
    let wallet_public_key = match get_bot_solana_wallet(&state.db, &config.id).await {
        Ok(Some(wallet)) => wallet.public_key,
        _ => std::env::var("SOLANA_WALLET_PUBLIC_KEY")
            .unwrap_or_else(|_| "DefaultPublicKey".to_string()),
    };
    
    // Start a separate agent for each token
    for token_address in &config.symbols {
        // Skip SOL and USDC, as they're not memecoins
        if token_address == SOL_MINT_ADDRESS || token_address == USDC_MINT_ADDRESS {
            continue;
        }
        
        let agent_state = state.clone();
        let agent_config = config.clone();
        let agent_wallet_key = wallet_public_key.clone();
        let token_mint = token_address.clone();
        
        tokio::spawn(async move {
            // Initialize RL agent
            let mut rl_agent = match RLTradingAgent::new(
                agent_state.clone(), 
                agent_config.id.clone(), 
                agent_wallet_key,
                token_mint.clone()
            ) {
                Ok(agent) => agent,
                Err(e) => {
                    tracing::error!("Failed to initialize RL agent for {}: {:?}", token_mint, e);
                    return;
                }
            };
            
            // Try to load existing model
            if let Err(e) = rl_agent.load_model().await {
                tracing::warn!("Could not load model for {}: {:?}", token_mint, e);
            }
            
            // Update initial portfolio state
            if let Err(e) = rl_agent.update_portfolio_from_onchain().await {
                tracing::error!("Failed to update portfolio data: {:?}", e);
            }
            
            // Initialize with market data
            for _ in 0..5 {
                if let Err(e) = rl_agent.fetch_market_data().await {
                    tracing::error!("Failed to fetch initial market data: {:?}", e);
                } else {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
            
            // Training loop
            let mut interval = interval(Duration::from_secs(agent_config.volatility_window as u64));
            let mut save_counter = 0;
            
            loop {
                interval.tick().await;
                
                // Check if we should execute trades based on market conditions
                let should_execute = match should_execute(&agent_state, &agent_config).await {
                    Some(value) => value,
                    None => continue,
                };
                
                if !should_execute {
                    continue;
                }
                
                // Execute a training step
                if let Err(e) = rl_agent.train_step().await {
                    tracing::error!("Error in RL training step: {:?}", e);
                    continue;
                }
                
                // Periodically save the model
                save_counter += 1;
                if save_counter >= 10 {
                    if let Err(e) = rl_agent.save_model().await {
                        tracing::error!("Failed to save model: {:?}", e);
                    }
                    save_counter = 0;
                }
                
                // Update metrics
                let token_balance = rl_agent.state.token_balance;
                let sol_balance = rl_agent.state.sol_balance;
                let portfolio_value = rl_agent.state.position_value;
                
                tracing::info!(
                    "Portfolio status for {}: SOL={}, Tokens={}, Value={}", 
                    token_mint, sol_balance, token_balance, portfolio_value
                );
                
                profit_gauge.record(portfolio_value, &[
                    KeyValue::new("bot_id", agent_config.id.clone()),
                    KeyValue::new("token", token_mint.clone()),
                ]);
            }
        });
    }
    
    // Main strategy loop just keeps track of overall metrics
    loop {
        interval.tick().await;
        
        // The actual trading is handled by the per-token agents
        tracing::debug!("RL Solana memecoin strategy main loop tick");
    }
}