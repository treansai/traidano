use crate::bot::bot_manager::BotManager;
use crate::configuration::BaseConfig;
use crate::core::rate_limiter::RateLimiter;
use crate::error::Error;
use crate::error::RequestError;
use axum::body::Body;
use http_body_util::BodyExt;
use hyper::{Method, Request};
use hyper_tls::HttpsConnector;
use opentelemetry::metrics::Meter;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::sync::Arc;
use tokio::sync::Mutex;
use traidano::RequestType;

#[derive(Debug)]
pub struct ClientBuildError;

/// Configuration of the api
///
#[derive(Debug, Clone, Deserialize, Serialize, Eq, PartialEq)]
pub struct ApiConfig {
    pub base_url: String,
    pub stream_url: String,
    pub stock_data_url: String,
    pub crypto_data_url: String,
    pub api_key: String,
    pub secret_key: String,
    pub jupiter_api_quote_url: String,
    pub jupiter_api_swap_url: String,
    pub jupiter_api_token_url: String,
    // social media urls
    pub twitter_url: Option<String>,
    pub telegram_url: Option<String>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            base_url: "base_url".to_string(),
            stream_url: "stream_url".to_string(),
            stock_data_url: "stock_data_url".to_string(),
            crypto_data_url: "crypto_data_url".to_string(),
            api_key: "api_key".to_string(),
            secret_key: "secret_key".to_string(),
            jupiter_api_quote_url: "jupiter_api_quote_url".to_string(),
            jupiter_api_swap_url: "jupiter_api_swap_url".to_string(),
            jupiter_api_token_url: "jupiter_api_token_url".to_string(),
            twitter_url: None,
            telegram_url: None,
        }
    }
}

impl ApiConfig {
    pub fn from_base_conf(base_config: BaseConfig) -> Self {
        Self {
            base_url: base_config.api_config.base_url,
            stream_url: base_config.api_config.stream_url,
            stock_data_url: base_config.api_config.stock_data_url,
            crypto_data_url: base_config.api_config.crypto_data_url,
            api_key: base_config.api_config.api_key.unwrap(),
            secret_key: base_config.api_config.secret.unwrap(),
            jupiter_api_quote_url: base_config.api_config.jupiter_api_quote_url,
            jupiter_api_swap_url: base_config.api_config.jupiter_api_swap_url,
            jupiter_api_token_url: base_config.api_config.jupiter_api_token_url,
            twitter_url: base_config.api_config.twitter_url,
            telegram_url: base_config.api_config.telegram_url,
        }
    }

    pub fn from_env_vars() -> ApiConfig {
        let base_url = std::env::var("BASE_URL").expect("BASE_URL is not set");
        let stream_url = std::env::var("STREAM_URL").expect("STREAM_URL is not set");
        let stock_data_url = std::env::var("STOCK_DATA_URL").expect("STOCK_DATA_URL is not set");
        let crypto_data_url = std::env::var("CRYPTO_DATA_URL").expect("CRYPTO_DATA_URL is not set");
        let api_key = std::env::var("API_KEY").expect("API_KEY is not set");
        let secret_key = std::env::var("SECRET_KEY").expect("SECRET_KEY is not set");
        let jupiter_api_quote_url = std::env::var("JUPITER_API_QUOTE_URL").expect("JUPITER_API_QUOTE_URL is not set");
        let jupiter_api_swap_url = std::env::var("JUPITER_API_SWAP_URL").expect("JUPITER_API_SWAP_URL is not set");
        let jupiter_api_token_url = std::env::var("JUPITER_API_TOKEN_URL").expect("JUPITER_API_TOKEN_URL is not set");
        let twitter_url = std::env::var("TWITTER_URL").or_else(|| None);
        let telegram_url = std::env::var("TELEGRAM_URL").or_else(|| None);
        
        Self {
            base_url,
            stream_url,
            stock_data_url,
            crypto_data_url,
            api_key,
            secret_key,
            jupiter_api_quote_url,
            jupiter_api_swap_url,
            jupiter_api_token_url,
            twitter_url: Some(twitter_url),
            telegram_url: Some(telegram_url),
        }
    }
}

pub struct Client {
    pub api_config: ApiConfig,
}

pub struct ClientBuilder {
    config: Option<ApiConfig>,
}

impl ClientBuilder {
    pub fn new() -> Self {
        Self { config: None }
    }

    pub fn config(mut self, config: ApiConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn build(self) -> Result<Client, ClientBuildError> {
        let config = match self.config {
            Some(conf) => conf,
            None => {
                tracing::error!("Base url required");
                return Err(ClientBuildError);
            }
        };

        Ok(Client { api_config: config })
    }
}

impl Client {
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    pub async fn send<T>(
        &self,
        method: Method,
        path: &str,
        body: Body,
        request_type: RequestType,
    ) -> Result<T, RequestError>
    where
        T: DeserializeOwned,
    {
        use hyper_util::{client::legacy::Client, rt::TokioExecutor};
        let https = HttpsConnector::new();
        let client = Client::builder(TokioExecutor::new()).build(https);

        // get the right url

        let mut full_url = self.url_match(request_type);
        full_url.push_str(path);

        let req = Request::builder()
            .method(method)
            .uri(&full_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("APCA-API-KEY-ID", &self.api_config.api_key)
            .header("APCA-API-SECRET-KEY", &self.api_config.secret_key)
            .body(body)
            .map_err(RequestError::HttpBuild)?;

        tracing::debug!("request  send : {:?}", req);
        let res = client
            .request(req)
            .await
            .map_err(RequestError::LegacyHyper)?;

        tracing::debug!("Response status: {}", res.status());
        tracing::debug!("Response: {:#?}\n", res);

        if res.status().is_success() {
            let body_bytes = res
                .into_body()
                .collect()
                .await
                .map_err(RequestError::Hyper)?
                .to_bytes();

            let response: T = serde_json::from_slice(&body_bytes)
                .map_err(|e| RequestError::Json(Error::Json(e)))?;
            Ok(response)
        } else {
            Err(RequestError::ApiError(res.status()))
        }
    }

    fn url_match(&self, request_type: RequestType) -> String {
        match request_type {
            RequestType::CryptoData => self.api_config.crypto_data_url.clone(),
            RequestType::StockData => self.api_config.stock_data_url.clone(),
            RequestType::Order => self.api_config.base_url.clone(),
        }
    }
}

pub struct AppState {
    pub alpaca_client: Client,
    pub db: PgPool,
    pub bot_manager: Mutex<BotManager>,
    pub rate_limiter: Arc<Mutex<RateLimiter>>,
    //pub tracer : BoxedTracer,
    pub meter: Meter,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum_macros::FromRequest;
    use mockito;
    use serde::{Deserialize, Serialize};

    #[derive(Deserialize, Debug, Serialize, PartialEq, Eq)]
    struct TestResponse {
        message: String,
    }

    #[test]
    fn create_client() {
        let api_config = ApiConfig {
            base_url: "base".to_string(),
            stream_url: "".to_string(),
            stock_data_url: "".to_string(),
            crypto_data_url: "".to_string(),
            api_key: "key".to_string(),
            secret_key: "secret".to_string(),
            jupiter_api_quote_url: "".to_string(),
            jupiter_api_swap_url: "".to_string(),
            jupiter_api_token_url: "".to_string(),
            twitter_url: None,
            telegram_url: None,
        };

        let client = Client::builder().config(api_config.clone()).build();

        assert_eq!(client.unwrap().api_config, api_config)
    }

    #[tokio::test]
    async fn test_send_success() {
        let mut mock_server = mockito::Server::new();
        let mock_url = mock_server.url();

        let api_config = ApiConfig {
            base_url: mock_url,
            ..ApiConfig::default()
        };

        // mock successful response
        let _m = mock_server
            .mock("GET", "/test-endpoint")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{'message': 'Success'"#)
            .create();

        let client = Client { api_config };
        let res: Result<TestResponse, RequestError> = client
            .send(
                Method::GET,
                "/test-endpoint",
                Body::empty(),
                RequestType::Order,
            )
            .await;

        assert!(res.is_ok());
        assert_eq!(
            res.unwrap(),
            TestResponse {
                message: "Success".to_string()
            }
        );
    }
}
