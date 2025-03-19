from ic.client import Client
from ic.identity import Identity
from ic.agent import Agent
from ic.candid import encode, Types

# Tworzenie instancji klienta i tożsamości
client = Client("http://127.0.0.1:4943/")
identity = Identity()

# Tworzenie agenta
agent = Agent(identity, client)

# Przykładowe zapytanie
canister_id = "bkyz2-fmaaa-aaaaa-qaaaq-cai"
response = agent.update_raw(canister_id, "load_model", encode([]))
# Wykonanie zapytania
# response = agent.update_raw(canister_id, "start_prompt", encode([{"type": Types.Text, "value": "Where is Poland?"}]),
#                             return_type=Types.Empty)
print("Odpowiedź:", response)
# response = agent.update_raw(canister_id, "start", encode([]))
# print("Odpowiedź:", response)

