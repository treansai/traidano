use crate::models::account::Account;

pub fn calculate_position_size(account: &Account, current_price: f64, risk_per_trade: f64) -> f64 {
    let risk_amount = account.equity * risk_per_trade;
    let shares = (risk_amount / current_price).floor();
    shares.min(account.buying_power / current_price)
}

pub fn calculate_stop_loss(current_price: f64, stop_loss_percentage: f64) -> f64 {
    current_price * (1.0 - stop_loss_percentage / 100.0)
}

pub fn calculate_take_profit(current_price: f64, take_profit_percentage: f64) -> f64 {
    current_price * (1.0 + take_profit_percentage / 100.0)
}

pub fn calculate_risk_per_trade(account: &Account, risk_per_trade: f64) -> f64 {
    account.equity * risk_per_trade
}

