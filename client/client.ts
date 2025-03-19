import "dotenv/config"
import {canisterId, createActor} from "../src/declarations/llama3_test_backend";
import {
    _SERVICE,
    SerializedModelRun,
    SerializedPromptSession,
    SerializedTensor
} from "../src/declarations/llama3_test_backend/llama3_test_backend.did";
import {ActorSubclass} from "@dfinity/agent";
import _ from 'lodash';

const host =
    process.env.DFX_NETWORK === "local" ? "http://localhost:4943" : undefined;

const prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" +
    "You are a helpful assistant<|eot_id|>\n" +
    "<|start_header_id|>user<|end_header_id|>\n" +
    "\n" +
    "Where is Poland placed?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

// const prompt = "<|start_header_id|>user<|end_header_id|>\n" +
//     "Where is Poland placed?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"


// const prompt = "Where is Poland placed?";
const main = async () => {
    const backend = createActor(canisterId!, {agentOptions: { host: host },});
    let session = await backend.start_prompt(prompt);
    // let result = await backend.start(session);
    let result = await start_iterative(backend, session);
    // let result = await start(backend, session);
    let text = result.at(0) as string;
    session = result.at(1) as SerializedPromptSession;
    console.log(text);
    for (let i = 0; i < 100; i++)
    {
        result = await step(backend, session);
        // result = await backend.step()
        text = result.at(0) as string;
        session = result.at(1) as SerializedPromptSession;
        console.log(text);
        // console.log(session);
    }
}

async function start(backend: ActorSubclass<_SERVICE>, session: SerializedPromptSession): Promise<[string, SerializedPromptSession]>
{
    let begin_result = await backend.begin_start(session, false);
    let run;
    if (begin_result[0].length > 0){
        run = begin_result[0][0];
    }
    else {
        console.error("run empty")
    }
    let result = await forward(backend, session, run, 1);
    session = result.at(0) as SerializedPromptSession;
    run = result.at(1) as SerializedModelRun;
    return await backend.end_step(run, session);
}

async function start_iterative(backend: ActorSubclass<_SERVICE>, session: SerializedPromptSession): Promise<[string, SerializedPromptSession]>
{
    let finished = false;
    let text;
    while (!finished) {
        let begin_result = await backend.begin_start(session, true);
        let run_option = begin_result.at(0) as SerializedModelRun | [];
        if (is_empty(run_option))
            return ["", session];
        let run = run_option[0] as SerializedModelRun;
        session = begin_result.at(1) as SerializedPromptSession;
        let forward_result = await forward(backend, session, run, 15);
        session = forward_result.at(0) as SerializedPromptSession;
        run = forward_result.at(1) as SerializedModelRun;
        let end_result = await backend.end_start(run, session);
        text = end_result.at(0);
        session = end_result.at(1) as SerializedPromptSession;
        console.log(`${session.tos.prompt_index} of ${session.tos.prompt.length}`);
        finished = text.length > 0;
    }
    return [text[0], session]
}


async function step(backend: ActorSubclass<_SERVICE>, session: SerializedPromptSession): Promise<[string, SerializedPromptSession]>
{
    let run = await backend.begin_step(getSimpleSession(session));
    let result = await forward(backend, session, run, 6);
    session = result[0];
    run = result[1];
    let endResult = await backend.end_step(run, getSimpleSession(session));
    let text = endResult[0];
    let newSession = endResult[1];
    session = updateSession(newSession, session);
    return [text, session];
}

async function forward(backend: ActorSubclass<_SERVICE>, session: SerializedPromptSession, run: SerializedModelRun, n:number): Promise<[SerializedPromptSession, SerializedModelRun]>
{
    let finished = false;
    while (!finished) {
        let reduced_session = get_reduced_session(session, run, BigInt(n));
        let result = await backend.forward(n, run, reduced_session);
        finished = result.at(0) as boolean;
        run = result.at(1) as SerializedModelRun;
        let newSession = result.at(2) as SerializedPromptSession;
        session = updateSession(session, newSession);
    }
    return [session, run]
}

function get_reduced_session(session: SerializedPromptSession, run: SerializedModelRun, steps_n: bigint): SerializedPromptSession
{
    let reduced_session : SerializedPromptSession = _.cloneDeep(session);
    if('Steps' in run.state)
    {
        let state = run.state as { 'Steps': bigint };
        let current_step = state.Steps;

        reduced_session.k_v_caches = reduced_session.k_v_caches.filter(x => {
            let layer = x[0] as bigint;

            return layer >= current_step && layer < current_step + steps_n;
        });
    }
    else {
        reduced_session.k_v_caches = [];
    }

    return reduced_session;
}

function getSimpleSession(session: SerializedPromptSession)
{
    let simpleSession : SerializedPromptSession = _.cloneDeep(session);
    simpleSession.k_v_caches = []
    return simpleSession;
}

function updateSession(base: SerializedPromptSession, update: SerializedPromptSession)
{
    const kv_map = base.k_v_caches.reduce((acc, [key, value]) => {
        acc[Number(key)] = value;
        return acc;
    }, {});
    update.k_v_caches.forEach(([key, value]) => kv_map[Number(key)] = value);
    base.k_v_caches = Object.entries(kv_map).map(
        ([key, value]) => [BigInt(key), value as [SerializedTensor, SerializedTensor]]
    );
    return base;
}

function is_empty(x: any)
{
    return Array.isArray(x) && x.length === 0;
}

main()
