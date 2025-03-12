mod internal;

use std::env;
use std::collections::HashMap;
use std::fmt::format;
use ic_cdk;

use candid::{CandidType, Principal};
use serde::{Deserialize, Serialize};
use strum_macros::Display;


#[derive(CandidType, Serialize, Deserialize, Debug, Display)]
pub enum Role {
    #[serde(rename = "system")]
    #[strum(serialize = "system")]
    System,

    #[serde(rename = "user")]
    #[strum(serialize = "user")]
    User,

    #[serde(rename = "assistant")]
    #[strum(serialize = "assistant")]
    Assistant,
}

#[derive(CandidType, Serialize, Deserialize, Debug)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ToString for ChatMessage {
    fn to_string(&self) -> String {
        let role = self.role.to_string();
        let content = &self.content;
        format!("<|start_header_id|>{role}<|end_header_id|>{content}<|eot_id|>")
    }
}

pub async fn prompt<P: ToString>(prompt_str: P) -> String {

    let messages = vec![
        ChatMessage {role: Role::System,
            content: "You are a helpful assistant. Respond using one sentence".to_string()},
        ChatMessage {role: Role::User, content: prompt_str.to_string()}
    ];
    chat(messages).await
}

pub async fn chat(messages: Vec<ChatMessage>) -> String {

    let mut string_messages = messages.iter().map(|m| m.to_string()).collect::<String>();
    let prompt =
        format!("<|begin_of_text|>{string_messages}<|start_header_id|>assistant<|end_header_id|>");

    ic_cdk::println!("{prompt}");
    internal::prompt(prompt).await
}
//
// let prompt = format!("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\
// You are a helpful assistant. Respond using one sentence<|eot_id|>\n\
// <|start_header_id|>user<|end_header_id|>\n\
// {prompt_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>");