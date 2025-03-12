use std::env;
use std::collections::HashMap;
use ic_cdk;

use candid::{CandidType, Principal};
use serde::{Deserialize, Serialize};

// --- Types --- //

#[derive(Clone, Debug, CandidType, Deserialize)]
enum TensorData {
    F32(Vec<f32>),
    U8(Vec<u8>),
}
#[derive(CandidType, Deserialize, Clone, Debug)]
struct SerializedTensor {
    data: TensorData,
    shape: Vec<usize>,
}


#[derive(CandidType, Deserialize, Clone)]
struct SerializedPromptSession {
    serialized_logit_processor: SerializedLogitProcessor,
    tos: SerializedTokenOutputStream,
    pub k_v_caches: HashMap<usize, (SerializedTensor, SerializedTensor)>

}

#[derive(CandidType, Deserialize, Clone)]
struct SerializedTokenOutputStream {
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
    prompt_index: usize,
    prompt: Vec<u32>
}

#[derive(CandidType, Deserialize, Clone)]
struct SerializedLogitProcessor {
    rng: String,
    sampling: Sampling,
}

#[derive(Clone, PartialEq, Debug, Deserialize, CandidType)]
enum Sampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
}

#[derive(CandidType, Deserialize, PartialEq, Clone)]
enum ModelRunState {
    Steps (usize),
    Finish,
    Finished,
}

#[derive(CandidType, Deserialize, Clone)]
struct SerializedModelRun {
    index_pos: usize,
    layer_in : SerializedTensor,
    mask : Option<SerializedTensor>,
    seq_len: usize,
    state: ModelRunState,
}


/// Starts the model run in a non-iterative fashion.
// async fn start(backend: &Principal, session: SerializedPromptSession)
//                -> (String, SerializedPromptSession)
// {
//     let (runs, session) = backend.begin_start(session, false).await;
//     let run = if !runs.is_empty() {
//         runs[0].clone()
//     } else {
//         eprintln!("run empty");
//         panic!("run empty");
//     };
//
//     let (session, run) = forward(backend, session, run, 1).await;
//     backend.end_step(run, session).await
// }


async fn start_iterative(backend: &Principal, mut session: SerializedPromptSession)
                         -> (String, SerializedPromptSession)
{
    let mut finished = false;
    let mut text = String::new();
    while !finished {
        let (run, new_session) : (Option<SerializedModelRun>, SerializedPromptSession)=
            ic_cdk::call(backend.clone(), "begin_start",(session.clone(), true)).await.unwrap();
        session = new_session;
        let run = run.unwrap();

        let (new_session, run) = forward(backend, session, run, 10).await;
        session = new_session;
        // let (texts, new_session) = backend.end_start(run, session).await;
        let (start_text, new_session) : (Option<String>, SerializedPromptSession) =
            ic_cdk::call(backend.clone(), "end_start",(run, session.clone())).await.unwrap();
        session = new_session;
        ic_cdk::println!("{} of {}", session.tos.prompt_index, session.tos.prompt.len());

        if start_text.is_some() {
            let start_text = start_text.unwrap();
            text=start_text;
            finished = true;
        }
    }
    (text, session)
}

/// Performs a single step.
async fn step(backend: & Principal, session: SerializedPromptSession)
              -> (Option<String>, bool, SerializedPromptSession)
{
    // let run = backend.begin_step(simple_session.clone()).await;
    let (run,) : (SerializedModelRun,) =
        ic_cdk::call(backend.clone(), "begin_step",(get_simple_session(&session),)).await.unwrap();
    let (session, run) = forward(backend, session, run, 5).await;
    // let (text, new_session) = backend.end_step(run, get_simple_session(&session)).await;
    let (text, eos, new_session) =
        ic_cdk::call(backend.clone(), "end_step",(run, get_simple_session(&session),)).await.unwrap();
    let updated_session = update_session(new_session, session);
    (text, eos, updated_session)
}

/// Runs the model forward for a given number of steps.
async fn forward(
    backend: &Principal,
    mut session: SerializedPromptSession,
    mut run: SerializedModelRun,
    n: usize,
) -> (SerializedPromptSession, SerializedModelRun)
{
    let mut finished = false;
    while !finished {
        let reduced_session = get_reduced_session(&session, &run, n);
        // let (fin, new_run, new_session) = backend.forward(n, run, reduced_session).await;
        let (fin, new_run, new_session) =
            ic_cdk::call(backend.clone(), "forward",(n as u8, run, reduced_session.clone(),)).await.unwrap();

        finished = fin;
        run = new_run;
        session = update_session(session, new_session);
    }
    (session, run)
}

/// Returns a reduced session containing only k_v_caches for layers in a given range.
fn get_reduced_session(
    session: &SerializedPromptSession,
    run: &SerializedModelRun,
    steps_n: usize,
) -> SerializedPromptSession {
    let mut reduced_session = session.clone();
    match run.state {
        ModelRunState::Steps(current_step) => {
            reduced_session.k_v_caches = reduced_session
                .k_v_caches
                .into_iter()
                .filter(|(layer, _)| *layer >= current_step && *layer < current_step + steps_n)
                .collect();
        }
        _ => {
            reduced_session.k_v_caches.clear();
        }
    }
    reduced_session
}

/// Returns a session with an empty k_v_caches.
fn get_simple_session(session: &SerializedPromptSession) -> SerializedPromptSession {
    let mut simple = session.clone();
    simple.k_v_caches.clear();
    simple
}

/// Merges the k_v_caches from two sessions.
fn update_session(
    mut base: SerializedPromptSession,
    update: SerializedPromptSession,
) -> SerializedPromptSession {
    let mut kv_map: HashMap<usize, (SerializedTensor, SerializedTensor)> = HashMap::new();
    for (key, value) in base.k_v_caches.into_iter() {
        kv_map.insert(key, value);
    }
    for (key, value) in update.k_v_caches.into_iter() {
        kv_map.insert(key, value);
    }
    base.k_v_caches = kv_map.into_iter().collect();
    base
}



pub async fn prompt<P: ToString>(prompt_str: P) -> String {
    let backend = Principal::from_text("be2us-64aaa-aaaaa-qaabq-cai").expect("invalid canister id");;

    let prompt_str = prompt_str.to_string();
    let mut result= String::new();
    let mut session: (SerializedPromptSession,) =
        ic_cdk::call(backend, "start_prompt",(prompt_str,)).await.unwrap();
    let mut session = session.0;
    let (mut text, mut session) = start_iterative(&backend, session).await;
    ic_cdk::println!("{}", text);
    result.push_str(&text);
    for _ in 0..50 {
        let (step_text, eos, new_session) = step(&backend, session).await;
        session = new_session;
        match step_text {
            Some(text) => {
                result.push_str(&text);
                ic_cdk::println!("{}", text);
            },
            None => {}
        }
        if eos
        {
            break;
        }

    }
    result
}

