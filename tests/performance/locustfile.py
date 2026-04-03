from locust import HttpUser, task, between
import json

class AuraUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def check_health(self):
        self.client.get("/api/health")

    @task(1)
    def chat_message(self):
        # We simulate a cognitive prompt, expecting the system to respond without dying.
        payload = {"message": "Hello Aura, what is your current CPU utilization?"}
        headers = {"Content-Type": "application/json"}
        self.client.post("/api/chat", data=json.dumps(payload), headers=headers)
