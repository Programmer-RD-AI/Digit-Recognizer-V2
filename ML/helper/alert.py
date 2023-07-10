from ML import *


class Alert:

    def __init__(self, results, past_results):
        self.results = results
        self.past_results = past_results

    def alert(self):
        msges = ""
        for key in self.results.keys():
            v = self.results[key]
            v_p = self.past_results[key]
            if "loss" in key.split() and v > v_p:
                msges += f"{key} has lowered"
            else:
                msges += f"{key} has increased"
        return msges
