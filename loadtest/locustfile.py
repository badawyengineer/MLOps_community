"""Load test for /ask. Handbook target: 50 concurrent users.

    locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 -t 60s \
        --csv reports/loadtest/ask --html reports/loadtest/ask.html

The mix is deliberate: Arabic and English questions, plus out-of-scope ones (the abstain path)
and the streaming endpoint, weighted roughly like real traffic.
"""

import random

from locust import HttpUser, between, task

ARABIC = ["هل يجوز تعديل العقد", "ما حكم الخطأ الذي يسبب ضررا للغير", "بماذا يحكم القاضي إذا لم يوجد نص تشريعي؟"]
ENGLISH = ["Must contracts be performed in good faith?", "Who must pay compensation for damage caused by a fault?"]
OUT_OF_SCOPE = ["ما عقوبة السرقة؟", "What is the penalty for drunk driving?"]


class AskUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(5)
    def ask_arabic(self):
        self.client.post("/ask", json={"question": random.choice(ARABIC)}, name="/ask [ar]")

    @task(3)
    def ask_english(self):
        self.client.post("/ask", json={"question": random.choice(ENGLISH)}, name="/ask [en]")

    @task(1)
    def ask_out_of_scope(self):
        self.client.post("/ask", json={"question": random.choice(OUT_OF_SCOPE)}, name="/ask [abstain]")

    @task(1)
    def ask_stream(self):
        with self.client.post("/ask/stream", json={"question": random.choice(ENGLISH)}, name="/ask/stream", stream=True, catch_response=True) as r:
            body = r.text
            if '"type": "done"' not in body:
                r.failure("stream ended without a done event")
            else:
                r.success()

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
