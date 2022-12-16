import math
import numpy

def create_testing_selector(client_info=None):
    return _testing_selector(client_info)

class _testing_selector:
    def __init__(self, client_info):
        self.client_info = client_info
        self.client_ids = [idx for idx in range(len(client_info))]

    def selected_bound(self, target, capacity, clients_num, confidence=0.85):

        factor = (1.0 - 2.0 * clients_num / math.log(1 - math.pow(confidence, 1)) * (target / float(capacity)) ** 2)
        n = (clients_num + 1.0) / factor

        return n

    def select_by_deviation(self, target, capacity_range, clients_num,
            confidence=0.85, overcommit=1.05):

        n = self.selected_bound(target, capacity_range, clients_num, confidence)
        num = n * overcommit
        selected_clients = numpy.random.choice(self.client_ids, replacement=False, size=num)
        return selected_clients

