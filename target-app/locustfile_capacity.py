from locust import HttpUser, task, between

class CapacityUser(HttpUser):
    wait_time = between(0.01, 0.02)

    @task
    def work(self):
        self.client.get("/work?ms=50", name="/work")
