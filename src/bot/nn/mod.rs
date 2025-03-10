use anyhow::Ok;
use solana_client::rpc_client::RpcClient;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    pubkey::Pubkey,
    signature::{Keypair, Signer},
    system_instruction,
    transaction::Transaction,
};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};
use log::{info, error};

// Constants
const LEARNING_RATE: f64 = 0.0005;
const DISCOUNT_FACTOR: f64 = 0.95;
const EPSILON: f64 = 0.15;
const MAX_POSITION_SIZE: f64 = 0.05;
const TRADING_FEE: f64 = 0.003;
const SLIPPAGE_TOLERANCE: f64 = 0.05;
const RPC_URL: &str = "https://api.mainnet-beta.solana.com";
const JUPITER_QUOTE_API: &str = "https://quote-api.jup.ag/v6/quote";
const JUPITER_SWAP_API: &str = "https://quote-api.jup.ag/v6/swap";



#[derive(Debug, Clone, Serialize, Deserialize)]
struct MarketData {
    price: f32,
    volume: f32,
    timestamp: u64,
}

// Trading State
struct TradingState {
    main_crypto_balance: f32,
    memecoin_balance: f32,
    price_history: Vec<f32>,
    volume_history: Vec<f32>,
    position_value: f32,
    total_balance: f32,
}

enum TradingAction {
    Buy(f32),
    Sell(f32),
    Hold,
}

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
            .add(nn::linear(vs / "layer5", 64 ,3, Default::default()));
        Self { model }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        self.model.forward(xs)
    }
}

pub struct TradingError {
    message: String,
}

struct TradingAgent {
    client: Arc<RcpClient>,
    wallet: Keypair,
    state: TradingState,
    q_network: QNetwork,
    optimizer: Optimizer,
    device: Device,
    http_client: HttpClient,
}


impl TradingAgent {
    fn new(wallet: Keypair, token_mint: &str) -> Result<Self, TradingError> {
        let client = RpcClient::new(RpcClient::new_with_commitment(
            RPC_URL,
            CommitmentConfig::confirmed(),
        ));

        let vs = nn::VarStore::new(Device::Cpu);
        let q_network = QNetwork::new(&vs.root());
        let optimizer =  nn::Adam::new(&vs, LEARNING_RATE); // add error handling

        let state = TradingState {
            main_crypto_balance: 0.0,
            memecoin_balance: 0.0,
            price_history: Vec::new(),
            volume_history: Vec::new(),
            position_value: 0.0,
            total_balance: 0.0,
        };

        Ok(Self {
            client,
            wallet,
            state,
            q_network,
            optimizer,
            device: Device::Cpu,
            http_client: HttpClient::new(),
        })
    }

    pub fn get_state_tensor(&self) -> Tensor {
        let price_data: Vec<f32> = self.state.price_history.iter().map(|p| *p as f32).collect();
        let volume_data: Vec<f32> = self.state.volume_history.iter().map(|v| *v as f32).collect();
        let sentiment_data: Vec<f32> = self.state.sentiment_history.iter().map(|s| *s as f32).collect();

        Tensor::of_slice(&[price_data, volume_data, sentiment_data].concat())
        .view(&[-1, -1])
        .to_device(self.device)
    }

    pub fn select_action(&self, state: &Tensor, epsilon: f64) -> TradingAction {
        if rand::random::<f32>() < epsilon {
            rand::random::<i32>() % 3 - 1 
        } else {
            let q_values = self.q_network.forward(state);
            q_values.argmax(0, false).int64_value(&[])
        }
    }

    async fn fetch_jupiter_quote(&self, input_mint: &str, output_mint: &str, amount: u64) -> Result<f64, Box<dyn Error>> {
        let url = format!(
            "{}?inputMint={}&outputMint={}&amount={}&slippageBps={}",
            JUPITER_QUOTE_API,
            input_mint,
            output_mint,
            amount,
            (SLIPPAGE_TOLERANCE * 10000.0) as u64 // Convert to basis points
        );

        let response = self.http_client.get(&url).send().await?;
        let json: Value = response.json().await?;

        let out_amount = json["outAmount"]
            .as_str()
            .ok_or("No outAmount in response")?
            .parse::<u64>()?;
        let in_amount = amount as f64;
        let price = out_amount as f64 / in_amount;
        Ok(price)
    }
    
    async fn execute_jupiter_swap(&mut self, input_mint: &str, output_mint: &str, amount: u64) -> Result<(f64, f64), Box<dyn Error>> {
        let quote_url = format!(
            "{}?inputMint={}&outputMint={}&amount={}&slippageBps={}",
            JUPITER_QUOTE_API,
            input_mint,
            output_mint,
            amount,
            (SLIPPAGE_TOLERANCE * 10000.0) as u64
        );

        let quote_response = self.http_client.get(&quote_url).send().await?;
        let quote_json: Value = quote_response.json().await?;

        let swap_body = serde_json::json!({
            "userPublicKey": self.wallet.pubkey().to_string(),
            "quoteResponse": quote_json,
            "wrapAndUnwrapSol": true
        });

        let swap_response = self.http_client
            .post(JUPITER_SWAP_API)
            .json(&swap_body)
            .send()
            .await?;

        let swap_json: Value = swap_response.json().await?;
        let swap_tx = swap_json["swapTransaction"]
            .as_str()
            .ok_or("No swapTransaction in response")?;

        // Decode and sign the transaction
        let decoded_tx = base64::decode(swap_tx)?;
        let mut transaction: Transaction = bincode::deserialize(&decoded_tx)?;
        transaction.sign(&[&self.wallet], self.client.get_latest_blockhash()?);

        // Send transaction to Solana
        let signature = self.client.send_and_confirm_transaction(&transaction)?;
        info!("Swap executed, signature: {}", signature);

        let out_amount = quote_json["outAmount"]
            .as_str()
            .ok_or("No outAmount in response")?
            .parse::<u64>()?;
        Ok((amount as f64, out_amount as f64))
    }

    async fn execute_trade(&mut self, action: i64, amount: f64) -> Result<f64, Box<dyn Error>> {
        let sol_mint = "So11111111111111111111111111111111111111112"; // Wrapped SOL mint
        let token_mint = self.token_mint.to_string();
        let mut reward = 0.0;

        let amount_lamports = (amount * self.state.sol_balance * 1_000_000_000.0) as u64; // Convert to lamports

        match action {
            1 => {
                // Buy: SOL -> Token
                let (in_amount, out_amount) = self.execute_jupiter_swap(sol_mint, &token_mint, amount_lamports).await?;
                self.state.sol_balance -= in_amount / 1_000_000_000.0;
                self.state.token_balance += out_amount / 1_000_000_000.0;
                info!("Bought {} tokens for {} SOL", out_amount / 1_000_000_000.0, in_amount / 1_000_000_000.0);
            }
            -1 => {
                // Sell: Token -> SOL
                let token_amount_lamports = (amount * self.state.token_balance * 1_000_000_000.0) as u64;
                let (in_amount, out_amount) = self.execute_jupiter_swap(&token_mint, sol_mint, token_amount_lamports).await?;
                self.state.token_balance -= in_amount / 1_000_000_000.0;
                self.state.sol_balance += out_amount / 1_000_000_000.0;
                info!("Sold {} tokens for {} SOL", in_amount / 1_000_000_000.0, out_amount / 1_000_000_000.0);
            }
            _ => {} // Hold
        }

        // Update price history with latest quote
        let current_price = self.fetch_jupiter_quote(sol_mint, &token_mint, 1_000_000_000).await?;
        self.state.price_history.push(current_price);
        if self.state.price_history.len() > 20 {
            self.state.price_history.remove(0);
        }

        // Calculate reward
        let new_value = self.state.sol_balance + self.state.token_balance * current_price;
        reward = (new_value - self.state.position_value) / self.state.position_value;
        reward += self.state.sentiment_history.last().unwrap_or(&0.0) * 0.1;
        self.state.position_value = new_value;

        // Stop-loss
        if reward < -0.2 {
            info!("Stop-loss triggered, exiting position");
            self.state.token_balance = 0.0;
            self.state.sol_balance = new_value;
        }

        Ok(reward)
    }

    async fn fetch_market_data(&mut self) -> Result<(), Box<dyn Error>> {
        let sol_mint = "So11111111111111111111111111111111111111112";
        let token_mint = self.token_mint.to_string();
        let new_price = self.fetch_jupiter_quote(sol_mint, &token_mint, 1_000_000_000).await?;
        let new_volume = rand::random::<f64>() * 10000.0; // Placeholder

        let avg_volume = self.state.volume_history.iter().sum::<f64>() / self.state.volume_history.len() as f64;
        let sentiment = if new_volume > avg_volume * 2.0 {
            1.0
        } else if new_volume < avg_volume * 0.5 {
            -1.0
        } else {
            0.0
        };

        self.state.price_history.push(new_price);
        self.state.volume_history.push(new_volume);
        self.state.sentiment_history.push(sentiment);
        
        if self.state.price_history.len() > 20 {
            self.state.price_history.remove(0);
            self.state.volume_history.remove(0);
            self.state.sentiment_history.remove(0);
        }

        Ok(())
    }

    async fn train_step(&mut self) -> Result<(), Box<dyn Error>> {
        let state = self.get_state_tensor();
        let action = self.select_action(&state, EPSILON);
        
        let reward = self.execute_trade(action, MAX_POSITION_SIZE).await?;
        self.fetch_market_data().await?;

        let next_state = self.get_state_tensor();
        let q_values = self.q_network.forward(&state);
        let next_q_values = self.q_network.forward(&next_state);
        
        let target = Tensor::of_slice(&[reward as f32])
            + Tensor::of_slice(&[DISCOUNT_FACTOR as f32])
            * next_q_values.max_dim(1, false).0;
        
        let action_idx = (action + 1) as i64;
        let loss = q_values
            .select(1, &Tensor::of_slice(&[action_idx]))
            .mse_loss(&target, tch::Reduction::Mean);

        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();

        Ok(())
    }

    async fn run(&mut self) -> Result<(), Box<dyn Error>> {
        let mut interval = time::interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            if let Err(e) = self.train_step().await {
                error!("Error in training step: {}", e);
            }
            
            info!(
                "Portfolio - SOL: {}, Tokens: {}, Value: {}, Sentiment: {}",
                self.state.sol_balance,
                self.state.token_balance,
                self.state.position_value,
                self.state.sentiment_history.last().unwrap_or(&0.0)
            );
        }
    }
}

