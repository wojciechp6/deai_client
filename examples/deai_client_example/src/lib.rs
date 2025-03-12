use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::future;
use std::future::Future;
use std::pin::Pin;
use std::process::Output;
use candid::{CandidType, Principal};
use ic_cdk::api::call::CallResult;
use serde::{Deserialize, Serialize};
use deai_client;
use deai_client::{prompt, ChatMessage};

#[ic_cdk::update]
async fn prompt_ai(prompt: String) -> String
{
    deai_client::prompt(prompt).await
}

#[ic_cdk::update]
async fn chat_ai(messages: Vec<ChatMessage>) -> String
{
    deai_client::chat(messages).await
}


ic_cdk::export_candid!();
