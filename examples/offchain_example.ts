import { prompt } from "../client/client";
async function main() {
    const promptStr = "What is the capital of France? Where is it placed?";
    try {
        const response = await prompt(promptStr);
        console.log();
        console.log("Full response:", response);
    } catch (error) {
        console.error("Error:", error);
    }
}

main();