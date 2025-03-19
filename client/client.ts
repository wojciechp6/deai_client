import { internal_prompt } from './internal';

enum Role {
    System = "system",
    User = "user",
    Assistant = "assistant"
}

interface ChatMessage {
    role: Role;
    content: string;
}

// Convert a ChatMessage to string
function chatMessageToString(message: ChatMessage): string {
    const role = message.role;
    const content = message.content;
    return `<|start_header_id|>${role}<|end_header_id|>${content}<|eot_id|>`;
}

// Function to create the prompt
export async function prompt<P extends string>(promptStr: P): Promise<string> {
    const messages: ChatMessage[] = [
        { role: Role.System, content: "You are a helpful assistant. Respond using one sentence" },
        { role: Role.User, content: promptStr }
    ];
    return await chat(messages);
}

// Function to handle chat messages
export async function chat(messages: ChatMessage[]): Promise<string> {
    const stringMessages = messages.map(chatMessageToString).join("");
    const prompt_text = `<|begin_of_text|>${stringMessages}<|start_header_id|>assistant<|end_header_id|>`;

    return await internal_prompt(prompt_text);
}