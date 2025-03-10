use crate::base::AppState;
use crate::dao::solana::{get_bot_memecoin_trades, get_bot_solana_wallet, register_solana_wallet};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// Request models
#[derive(Debug, Deserialize)]
pub struct RegisterWalletRequest {
    pub public_key: String,
    pub encrypted_private_key: Option<String>,
    pub label: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TokenInfoResponse {
    pub symbol: String,
    pub address: String,
    pub price_usd: Option<f64>,
    pub volume_24h: Option<f64>,
    pub market_cap: Option<f64>,
    pub holders: Option<u32>,
    pub social_score: Option<f64>,
    pub is_trending: bool,
}

// Register a Solana wallet for a bot
pub async fn register_bot_wallet(
    State(state): State<Arc<AppState>>,
    Path(bot_id): Path<String>,
    Json(request): Json<RegisterWalletRequest>,
) -> Result<StatusCode, StatusCode> {
    match register_solana_wallet(
        &state.db,
        &bot_id,
        &request.public_key,
        request.encrypted_private_key.as_deref(),
        request.label.as_deref(),
    )
    .await
    {
        Ok(_) => Ok(StatusCode::CREATED),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

// Get a bot's Solana wallet
pub async fn get_wallet(
    State(state): State<Arc<AppState>>,
    Path(bot_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match get_bot_solana_wallet(&state.db, &bot_id).await {
        Ok(Some(wallet)) => {
            // Return wallet but exclude any sensitive fields
            let response = serde_json::json!({
                "id": wallet.id,
                "bot_id": wallet.bot_id,
                "public_key": wallet.public_key,
                "label": wallet.label,
                "created_at": wallet.created_at,
                "updated_at": wallet.updated_at,
            });
            Ok(Json(response))
        }
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

// Get all memecoin trades for a bot
pub async fn get_trades(
    State(state): State<Arc<AppState>>,
    Path(bot_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match get_bot_memecoin_trades(&state.db, &bot_id).await {
        Ok(trades) => {
            let response = serde_json::json!({
                "trades": trades,
                "count": trades.len(),
            });
            Ok(Json(response))
        }
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

// Get trending tokens from external APIs (Jupiter, PumpFun, etc.)
pub async fn get_trending_tokens(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<TokenInfoResponse>>, StatusCode> {
    // This would call out to external APIs to fetch trending tokens
    // For demonstration, returning placeholder data
    let trending_tokens = vec![
        TokenInfoResponse {
            symbol: "WIF".to_string(),
            address: "DogWifHat111111111111111111111111111111111".to_string(),
            price_usd: Some(0.85),
            volume_24h: Some(2500000.0),
            market_cap: Some(85000000.0),
            holders: Some(12500),
            social_score: Some(0.92),
            is_trending: true,
        },
        TokenInfoResponse {
            symbol: "BONK".to_string(),
            address: "BONK97iADxYQ2ndSjL5i3JT45M2restUZ3TEMJYXhF".to_string(),
            price_usd: Some(0.00002),
            volume_24h: Some(15000000.0),
            market_cap: Some(250000000.0),
            holders: Some(65000),
            social_score: Some(0.85),
            is_trending: true,
        },
    ];

    Ok(Json(trending_tokens))
}