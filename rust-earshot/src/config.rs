//! Configuration loading from YAML

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub audio: AudioConfig,
    pub vad: VadConfig,
    pub stt: SttConfig,
    pub tts: TtsConfig,
    pub llm: LlmConfig,
    #[serde(default)]
    pub notifications: NotificationConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NotificationConfig {
    /// Enable notification server
    #[serde(default = "default_notifications_enabled")]
    pub enabled: bool,
    /// TCP port for notification server
    #[serde(default = "default_notifications_port")]
    pub port: u16,
    /// Idle threshold for polling queue (seconds)
    #[serde(default = "default_idle_poll_threshold")]
    pub idle_poll_threshold_s: u64,
    /// Idle threshold for immediate delivery (seconds)
    #[serde(default = "default_idle_immediate_threshold")]
    pub idle_immediate_threshold_s: u64,
    /// Grace period to batch multiple notifications (milliseconds)
    #[serde(default = "default_batch_grace_period")]
    pub batch_grace_period_ms: u64,
    /// System prompt for notification delivery
    #[serde(default = "default_notification_prompt")]
    pub system_prompt: String,
}

fn default_notifications_enabled() -> bool { true }
fn default_notifications_port() -> u16 { 9999 }
fn default_idle_poll_threshold() -> u64 { 10 }
fn default_idle_immediate_threshold() -> u64 { 5 }
fn default_batch_grace_period() -> u64 { 700 }
fn default_notification_prompt() -> String {
    r#"You are a voice assistant. You have received a notification from another application that you need to relay to the user.

The notification details are provided below. Your job is to:
1. Tell the user about this notification in a natural, conversational way
2. Keep it brief - just the essential information
3. Speak as if you're a helpful assistant letting them know about something

Do NOT say things like "I don't see a notification" - the notification content IS the message you received."#.to_string()
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enabled: default_notifications_enabled(),
            port: default_notifications_port(),
            idle_poll_threshold_s: default_idle_poll_threshold(),
            idle_immediate_threshold_s: default_idle_immediate_threshold(),
            batch_grace_period_ms: default_batch_grace_period(),
            system_prompt: default_notification_prompt(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_channels")]
    pub channels: u16,
    #[serde(default = "default_frame_ms")]
    pub frame_ms: u32,
    #[serde(default)]
    pub input_device: Option<String>,
    #[serde(default)]
    pub output_device: Option<String>,
    #[serde(default = "default_list_devices")]
    pub list_devices: bool,
}

fn default_sample_rate() -> u32 { 16000 }
fn default_channels() -> u16 { 1 }
fn default_frame_ms() -> u32 { 30 }
fn default_list_devices() -> bool { false }

#[derive(Debug, Clone, Deserialize)]
pub struct VadConfig {
    #[serde(default = "default_model_path")]
    pub model_path: String,
    #[serde(default = "default_start_threshold")]
    pub start_threshold: f32,
    #[serde(default = "default_end_threshold")]
    pub end_threshold: f32,
    #[serde(default = "default_ema_alpha")]
    pub ema_alpha: f32,
    #[serde(default = "default_pre_ms")]
    pub pre_ms: u32,
    #[serde(default = "default_hang_ms")]
    pub hang_ms: u32,
    #[serde(default = "default_post_ms")]
    pub post_ms: u32,
    #[serde(default = "default_max_segment_s")]
    pub max_segment_s: f32,
    #[serde(default = "default_min_start_frames")]
    pub min_start_frames: u32,
    #[serde(default = "default_min_speech_ms")]
    pub min_speech_ms: u32,
}

fn default_model_path() -> String { "models/silero_vad.onnx".to_string() }
fn default_start_threshold() -> f32 { 0.35 }
fn default_end_threshold() -> f32 { 0.25 }
fn default_ema_alpha() -> f32 { 0.30 }
fn default_pre_ms() -> u32 { 600 }
fn default_hang_ms() -> u32 { 700 }
fn default_post_ms() -> u32 { 400 }
fn default_max_segment_s() -> f32 { 30.0 }
fn default_min_start_frames() -> u32 { 3 }
fn default_min_speech_ms() -> u32 { 700 }

#[derive(Debug, Clone, Deserialize)]
pub struct SttConfig {
    #[serde(default = "default_stt_url")]
    pub url: String,
    pub auth_token: Option<String>,
}

fn default_stt_url() -> String { "ws://localhost:8765".to_string() }

#[derive(Debug, Clone, Deserialize)]
pub struct TtsConfig {
    #[serde(default = "default_tts_url")]
    pub url: String,
    pub auth_token: Option<String>,
}

fn default_tts_url() -> String { "ws://localhost:8766".to_string() }

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    #[serde(default = "default_llm_url")]
    pub url: String,
    #[serde(default = "default_llm_model")]
    pub model: String,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
}

fn default_llm_url() -> String { "http://localhost:1234/v1/chat/completions".to_string() }
fn default_llm_model() -> String { "local-model".to_string() }
fn default_temperature() -> f32 { 0.7 }
fn default_system_prompt() -> String {
    "You are a helpful voice assistant. Keep responses concise and conversational.".to_string()
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: default_sample_rate(),
            channels: default_channels(),
            frame_ms: default_frame_ms(),
            input_device: None,
            output_device: None,
            list_devices: default_list_devices(),
        }
    }
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            model_path: default_model_path(),
            start_threshold: default_start_threshold(),
            end_threshold: default_end_threshold(),
            ema_alpha: default_ema_alpha(),
            pre_ms: default_pre_ms(),
            hang_ms: default_hang_ms(),
            post_ms: default_post_ms(),
            max_segment_s: default_max_segment_s(),
            min_start_frames: default_min_start_frames(),
            min_speech_ms: default_min_speech_ms(),
        }
    }
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            url: default_stt_url(),
            auth_token: None,
        }
    }
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            url: default_tts_url(),
            auth_token: None,
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            url: default_llm_url(),
            model: default_llm_model(),
            temperature: default_temperature(),
            system_prompt: default_system_prompt(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            audio: AudioConfig::default(),
            vad: VadConfig::default(),
            stt: SttConfig::default(),
            tts: TtsConfig::default(),
            llm: LlmConfig::default(),
            notifications: NotificationConfig::default(),
        }
    }
}

pub fn load_config(path: &str) -> Result<Config> {
    if std::path::Path::new(path).exists() {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path))?;
        let cfg: Config = serde_yaml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path))?;
        Ok(cfg)
    } else {
        // Use defaults if no config file
        Ok(Config::default())
    }
}
